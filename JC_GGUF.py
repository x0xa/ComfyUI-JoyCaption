import torch
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
import base64
import io
import sys
import gc
import os
import threading
import time
import random
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from huggingface_hub import hf_hub_download
from gguf_worker import GGUFWorkerProcess
from caption_sanitize import sanitize_caption
from image_utils import fit_contain
from server import PromptServer

_last_progress_time = 0
_progress_throttle_interval = 5.0

STOP_TOKENS = ["</s>", "User:", "Assistant:", "USER:", "ASSISTANT:"]


class ProgressNotifier:
    """Sends progress updates via WebSocket every N seconds"""

    def __init__(self, message: str, interval: float = 3.0):
        self.message = message
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self._send()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        return False

    def _send(self):
        global _last_progress_time
        current_time = time.time()
        if current_time - _last_progress_time < _progress_throttle_interval:
            return
        try:
            if hasattr(PromptServer, 'instance') and PromptServer.instance:
                PromptServer.instance.send_sync("progress", {"message": self.message})
                _last_progress_time = current_time
                print(f"[QwenCaption GGUF] {self.message}")
        except Exception as e:
            print(f"[QwenCaption GGUF] Failed to send progress: {e}")

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._send()


class ModelLoadError(Exception):
    pass


def suppress_output(func):
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return wrapper


os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    MODEL_SETTINGS = config["model_settings"]
    GGUF_SETTINGS = config["gguf_settings"]

_MODEL_CACHE = {}


def free_comfy_vram(reason=""):
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache(force=True)
    except Exception as e:
        print(f"[QwenCaption GGUF] Could not unload ComfyUI models: {e}")
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    print(f"[QwenCaption GGUF] Freed ComfyUI VRAM before loading model{(' (' + reason + ')') if reason else ''}")


def clear_model_cache():
    global _MODEL_CACHE
    for cache_key in list(_MODEL_CACHE.keys()):
        try:
            cached = _MODEL_CACHE[cache_key]
            if cached is not None and hasattr(cached, 'cleanup'):
                cached.cleanup()
        except Exception as e:
            print(f"[QwenCaption GGUF] Warning: Error clearing cache entry {cache_key}: {e}")
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[QwenCaption GGUF] Model cache cleared")


def resolve_gguf_paths():
    """Resolve the model + mmproj GGUF paths, downloading them on first use if missing."""
    llm_models_dir = (Path(folder_paths.models_dir).resolve() / "LLM" / "GGUF").resolve()
    llm_models_dir.mkdir(parents=True, exist_ok=True)

    def ensure(filename, repo):
        local_path = llm_models_dir / filename
        if local_path.exists():
            return local_path
        return Path(hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(llm_models_dir),
            local_dir_use_symlinks=False,
        )).resolve()

    model_path = ensure(GGUF_SETTINGS["model_filename"], GGUF_SETTINGS["model_repo"])
    mmproj_path = ensure(GGUF_SETTINGS["mmproj_filename"], GGUF_SETTINGS["mmproj_repo"])
    return model_path, mmproj_path


def resolve_gpu_layers(processing_mode):
    if processing_mode == "CPU":
        return 0
    if processing_mode == "GPU":
        return -1
    return -1 if torch.cuda.is_available() else 0


class QwenCaptionModel:
    """In-process GGUF model (Keep in Memory / Clear After Run / Global Cache)."""

    def __init__(self, processing_mode: str):
        self.model = None
        self.chat_handler = None
        try:
            with ProgressNotifier("Resolving model files..."):
                model_path, mmproj_path = resolve_gguf_paths()

            n_gpu_layers = resolve_gpu_layers(processing_mode)
            if n_gpu_layers != 0 and torch.cuda.is_available():
                with ProgressNotifier("Freeing ComfyUI VRAM..."):
                    free_comfy_vram("in-process loader")

            with ProgressNotifier("Initializing vision handler..."):
                self.chat_handler = self._load_clip_handler(mmproj_path)
            with ProgressNotifier("Loading model into memory..."):
                self.model = self._load_llama_model(model_path, n_gpu_layers)
        except Exception as e:
            raise ModelLoadError(f"Model initialization failed: {str(e)}")

    @suppress_output
    def _load_clip_handler(self, mmproj_path):
        return Qwen25VLChatHandler(clip_model_path=str(mmproj_path))

    @suppress_output
    def _load_llama_model(self, model_path, n_gpu_layers):
        return Llama(
            model_path=str(model_path),
            n_ctx=MODEL_SETTINGS["context_window"],
            n_batch=2048,
            n_threads=max(4, MODEL_SETTINGS["cpu_threads"]),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            chat_handler=self.chat_handler,
            offload_kqv=True,
            numa=True,
        )

    def generate(self, image, system, prompt, max_new_tokens, temperature, top_p, top_k):
        img_buffer = None
        response = None
        try:
            with ProgressNotifier("Processing image..."):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = fit_contain(image, GGUF_SETTINGS["image_size"])
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                data_uri = f"data:image/png;base64,{base64.b64encode(img_buffer.read()).decode('utf-8')}"

            params = {
                "messages": [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt.strip()},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]},
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": random.randint(1, 2 ** 31 - 1),
                "stop": STOP_TOKENS,
                "stream": False,
                "repeat_penalty": 1.1,
                "mirostat_mode": 0,
            }
            if top_k > 0:
                params["top_k"] = top_k

            with ProgressNotifier("Generating caption..."):
                response = self._create_completion(params)
                return sanitize_caption(response["choices"][0]["message"]["content"])
        finally:
            if img_buffer is not None:
                img_buffer.close()
                del img_buffer
            if response is not None:
                del response
            gc.collect()

    @suppress_output
    def _create_completion(self, params):
        return self.model.create_chat_completion(**params)

    def cleanup(self):
        try:
            if self.model is not None:
                self.model.chat_handler = None
                if hasattr(self.model, 'close'):
                    self.model.close()
                if hasattr(self.model, '_model') and self.model._model is not None:
                    self.model._model = None
                del self.model
                self.model = None
            self.chat_handler = None
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print("[QwenCaption GGUF] Cleanup completed")
        except Exception as e:
            print(f"[QwenCaption GGUF] Warning during cleanup: {e}")
            self.model = None
            self.chat_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class QwenCaptionModelSubprocess:
    """Subprocess-isolated GGUF model. Guarantees full memory release on cleanup()."""

    def __init__(self, processing_mode: str):
        self.worker = None
        self.image_size = GGUF_SETTINGS["image_size"]
        try:
            with ProgressNotifier("Resolving model files..."):
                model_path, mmproj_path = resolve_gguf_paths()

            n_gpu_layers = resolve_gpu_layers(processing_mode)
            if n_gpu_layers != 0 and torch.cuda.is_available():
                with ProgressNotifier("Freeing ComfyUI VRAM..."):
                    free_comfy_vram("subprocess loader")

            with ProgressNotifier("Starting isolated worker process..."):
                self.worker = GGUFWorkerProcess(
                    model_path=str(model_path),
                    mmproj_path=str(mmproj_path),
                    n_ctx=MODEL_SETTINGS["context_window"],
                    n_batch=2048,
                    n_threads=max(4, MODEL_SETTINGS["cpu_threads"]),
                    n_gpu_layers=n_gpu_layers,
                    timeout=180.0,
                )
            print("[QwenCaption GGUF] Subprocess worker ready")
        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Model initialization failed: {str(e)}")

    def generate(self, image, system, prompt, max_new_tokens, temperature, top_p, top_k):
        if self.worker is None or not self.worker.is_alive():
            raise RuntimeError("Worker process is not running")

        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        image_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        img_buffer.close()

        with ProgressNotifier("Generating caption..."):
            return self.worker.generate(
                image_b64=image_b64,
                system=system,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=self.image_size,
            )

    def cleanup(self):
        if self.worker is not None:
            self.worker.cleanup()
            self.worker = None
            print("[QwenCaption GGUF] Subprocess cleanup completed - memory fully released")


class QwenCaptionGGUF:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "max_new_tokens": ("INT", {"default": MODEL_SETTINGS["default_max_tokens"], "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": MODEL_SETTINGS["default_temperature"], "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": MODEL_SETTINGS["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": MODEL_SETTINGS["default_top_k"], "min": 0, "max": 100}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Clear After Run (Subprocess)", "Global Cache"], {"default": "Keep in Memory"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝Caption"

    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None

    def generate(self, image, processing_mode, max_new_tokens, temperature, top_p, top_k, custom_prompt, memory_management):
        try:
            cache_enabled = (memory_management == "Global Cache")
            use_subprocess = (memory_management == "Clear After Run (Subprocess)")

            if self.predictor is None or self.current_processing_mode != processing_mode:
                if self.predictor is not None:
                    self.predictor.cleanup()
                    self.predictor = None

                if cache_enabled and processing_mode in _MODEL_CACHE:
                    self.predictor = _MODEL_CACHE[processing_mode]
                else:
                    self.predictor = (QwenCaptionModelSubprocess(processing_mode)
                                      if use_subprocess else QwenCaptionModel(processing_mode))
                    if cache_enabled:
                        _MODEL_CACHE[processing_mode] = self.predictor
                self.current_processing_mode = processing_mode

            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            response = self.predictor.generate(
                image=pil_image,
                system=MODEL_SETTINGS["default_system_prompt"],
                prompt=custom_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)"):
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None

            return (custom_prompt, response)
        except Exception:
            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)") and self.predictor is not None:
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None
            raise


NODE_CLASS_MAPPINGS = {
    "QwenCaptionGGUF": QwenCaptionGGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenCaptionGGUF": "Qwen Caption (GGUF)",
}
