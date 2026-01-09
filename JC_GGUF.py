import torch
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import io
import sys
import gc
import os
from huggingface_hub import hf_hub_download
from gguf_worker import GGUFWorkerProcess

class ModelLoadError(Exception):
    pass

def suppress_output(func):
    import sys
    import io
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
    CAPTION_TYPE_MAP = config["caption_type_map"]
    EXTRA_OPTIONS = config["extra_options"]
    MODEL_SETTINGS = config["model_settings"]
    CAPTION_LENGTH_CHOICES = config["caption_length_choices"]
    GGUF_MODELS = config["gguf_models"]
    GGUF_SETTINGS = config["gguf_settings"]

# --- Custom Models Merge Logic (for GGUF models only) ---
custom_path = Path(__file__).parent / "custom_models.json"

if custom_path.exists():
    try:
        with open(custom_path, "r", encoding="utf-8") as f:
            custom_data = json.load(f) or {}
        GGUF_MODELS.update(custom_data.get("gguf_models", {}))
        print("[JoyCaption GGUF] ✅ Loaded custom GGUF custom models")
    except Exception as e:
        print(f"[JoyCaption GGUF] ⚠️ Failed to load custom models → {e}")
else:
    print("[JoyCaption GGUF] ℹ️ No custom models found, skipping user-defined GGUF models.")
# -------------------------------------------------------

_MODEL_CACHE = {}

def clear_model_cache():
    """Clears all cached GGUF models and frees memory."""
    global _MODEL_CACHE
    for cache_key in list(_MODEL_CACHE.keys()):
        try:
            cached = _MODEL_CACHE[cache_key]
            if cached is not None and hasattr(cached, 'cleanup'):
                cached.cleanup()
        except Exception as e:
            print(f"[JoyCaption GGUF] Warning: Error clearing cache entry {cache_key}: {e}")
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[JoyCaption GGUF] Model cache cleared")

def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2

    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    if extra_options:
        prompt += " " + " ".join(extra_options)
    return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)

class JC_GGUF_Models:
    def __init__(self, model: str, processing_mode: str):
        self.model = None
        self.chat_handler = None  # Keep separate reference for cleanup
        try:
            models_dir = Path(folder_paths.models_dir).resolve()
            llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
            llm_models_dir.mkdir(parents=True, exist_ok=True)

            model_filename = Path(model).name
            local_path = llm_models_dir / model_filename

            if not local_path.exists():
                if "/" not in model:
                    raise ValueError("Invalid model path")
                repo_path, filename = model.rsplit("/", 1)
                local_path = Path(hf_hub_download(
                    repo_id=repo_path,
                    filename=filename,
                    local_dir=str(llm_models_dir),
                    local_dir_use_symlinks=False
                )).resolve()

            mmproj_filename = GGUF_SETTINGS["mmproj_filename"]
            mmproj_path = llm_models_dir / mmproj_filename
            if not mmproj_path.exists():
                mmproj_path = Path(hf_hub_download(
                    repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
                    filename=mmproj_filename,
                    local_dir=str(llm_models_dir),
                    local_dir_use_symlinks=False
                )).resolve()

            n_ctx = MODEL_SETTINGS["context_window"]
            n_batch = 2048
            n_threads = max(4, MODEL_SETTINGS["cpu_threads"])
            if processing_mode == "Auto":
                n_gpu_layers = -1 if torch.cuda.is_available() else 0
            elif processing_mode == "GPU":
                n_gpu_layers = -1
            else:  # CPU
                n_gpu_layers = 0

            self.chat_handler, self.model = self._initialize_model(local_path, mmproj_path, n_ctx, n_batch, n_threads, n_gpu_layers)

        except Exception as e:
            raise ModelLoadError(f"Model initialization failed: {str(e)}")

    @suppress_output
    def _initialize_model(self, local_path, mmproj_path, n_ctx, n_batch, n_threads, n_gpu_layers):
        """Initialize the GGUF model with suppressed output"""
        chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
        model = Llama(
            model_path=str(local_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            chat_handler=chat_handler,
            offload_kqv=True,
            numa=True
        )
        return chat_handler, model

    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        img_buffer = None
        messages = None
        response = None
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize(GGUF_SETTINGS["default_image_size"], Image.Resampling.BILINEAR)

            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_base64}"

            messages = [
                {"role": "system", "content": system.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip()},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ]

            import random
            completion_params = {
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": random.randint(1, 2**31 - 1),  # Random seed for variety
                "stop": ["</s>", "User:", "Assistant:", "USER:", "ASSISTANT:", "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:", "ASISTANT\n", "ASISTANT:", "ASSENT", "ASSENTED"],
                "stream": False,
                "repeat_penalty": 1.1,
                "mirostat_mode": 0
            }

            if top_k > 0:
                completion_params["top_k"] = top_k

            response = self._create_completion(completion_params)
            result = response["choices"][0]["message"]["content"].strip()
            return result

        except Exception as e:
            return f"Generation error: {str(e)}"
        finally:
            # Clean up intermediate data to prevent memory buildup
            if img_buffer is not None:
                img_buffer.close()
                del img_buffer
            if messages is not None:
                del messages
            if response is not None:
                del response
            gc.collect()

    @suppress_output
    def _create_completion(self, completion_params):
        """Create chat completion with suppressed output"""
        return self.model.create_chat_completion(**completion_params)

    def cleanup(self):
        """Release model resources and free memory."""
        try:
            # First, free the CLIP model from chat_handler (this is the main memory leak!)
            # Use our saved reference which is more reliable
            chat_handler = self.chat_handler
            if chat_handler is None and self.model is not None and hasattr(self.model, 'chat_handler'):
                chat_handler = self.model.chat_handler

            if chat_handler is not None:
                # Explicitly free CLIP context if available
                if hasattr(chat_handler, 'clip_ctx') and chat_handler.clip_ctx is not None:
                    try:
                        if hasattr(chat_handler, '_clip_free') and chat_handler._clip_free is not None:
                            chat_handler._clip_free(chat_handler.clip_ctx)
                            print("[JoyCaption GGUF] CLIP model freed via _clip_free")
                        elif hasattr(chat_handler, '_llava_cpp') and chat_handler._llava_cpp is not None:
                            chat_handler._llava_cpp.clip_free(chat_handler.clip_ctx)
                            print("[JoyCaption GGUF] CLIP model freed via _llava_cpp.clip_free")
                    except Exception as e:
                        print(f"[JoyCaption GGUF] Warning: Could not free CLIP context: {e}")
                    chat_handler.clip_ctx = None

                # Clear chat_handler references
                if self.model is not None and hasattr(self.model, 'chat_handler'):
                    self.model.chat_handler = None
                self.chat_handler = None
                del chat_handler

            if self.model is not None:
                # Close the main Llama model
                if hasattr(self.model, 'close'):
                    self.model.close()
                    print("[JoyCaption GGUF] Llama model closed")

                # Clear any remaining references
                if hasattr(self.model, '_model') and self.model._model is not None:
                    self.model._model = None

                del self.model
                self.model = None

            # Force garbage collection multiple times to ensure cleanup
            gc.collect()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            gc.collect()
            print("[JoyCaption GGUF] Cleanup completed")

        except Exception as e:
            print(f"[JoyCaption GGUF] Warning during cleanup: {e}")
            # Try to force cleanup even on error
            self.model = None
            self.chat_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class JC_GGUF_Models_Subprocess:
    """
    GGUF model wrapper using subprocess isolation.
    Guarantees 100% memory release when cleanup() is called.
    """

    def __init__(self, model: str, processing_mode: str):
        self.worker = None
        self.image_size = tuple(GGUF_SETTINGS["default_image_size"])

        try:
            models_dir = Path(folder_paths.models_dir).resolve()
            llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
            llm_models_dir.mkdir(parents=True, exist_ok=True)

            model_filename = Path(model).name
            local_path = llm_models_dir / model_filename

            if not local_path.exists():
                if "/" not in model:
                    raise ValueError("Invalid model path")
                repo_path, filename = model.rsplit("/", 1)
                local_path = Path(hf_hub_download(
                    repo_id=repo_path,
                    filename=filename,
                    local_dir=str(llm_models_dir),
                    local_dir_use_symlinks=False
                )).resolve()

            mmproj_filename = GGUF_SETTINGS["mmproj_filename"]
            mmproj_path = llm_models_dir / mmproj_filename
            if not mmproj_path.exists():
                mmproj_path = Path(hf_hub_download(
                    repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
                    filename=mmproj_filename,
                    local_dir=str(llm_models_dir),
                    local_dir_use_symlinks=False
                )).resolve()

            n_ctx = MODEL_SETTINGS["context_window"]
            n_batch = 2048
            n_threads = max(4, MODEL_SETTINGS["cpu_threads"])
            if processing_mode == "Auto":
                n_gpu_layers = -1 if torch.cuda.is_available() else 0
            elif processing_mode == "GPU":
                n_gpu_layers = -1
            else:  # CPU
                n_gpu_layers = 0

            print("[JoyCaption GGUF] Starting subprocess worker...")
            self.worker = GGUFWorkerProcess(
                model_path=str(local_path),
                mmproj_path=str(mmproj_path),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                timeout=180.0
            )
            print("[JoyCaption GGUF] Subprocess worker ready")

        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Model initialization failed: {str(e)}")

    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int,
                 temperature: float, top_p: float, top_k: int) -> str:
        if self.worker is None or not self.worker.is_alive():
            return "Error: Worker process is not running"

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Encode image to base64
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            image_b64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            img_buffer.close()

            return self.worker.generate(
                image_b64=image_b64,
                system=system,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                image_size=self.image_size
            )
        except Exception as e:
            return f"Generation error: {str(e)}"

    def cleanup(self):
        """Terminate subprocess - guarantees 100% memory release."""
        if self.worker is not None:
            self.worker.cleanup()
            self.worker = None
            print("[JoyCaption GGUF] Subprocess cleanup completed - memory fully released")

class JC_GGUF:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always return NaN to force re-execution (no caching)
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto", "tooltip": "Auto: Automatically detect best mode\nGPU: Faster but requires more VRAM\nCPU: Slower but saves VRAM"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive", "tooltip": "Select the style of caption you want to generate"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any", "tooltip": "Control the length of the generated caption"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Clear After Run (Subprocess)", "Global Cache"], {"default": "Keep in Memory", "tooltip": "Choose how to manage model memory. 'Keep in Memory' for faster processing, 'Clear After Run' may leak ~850MB, 'Clear After Run (Subprocess)' guarantees 100% memory release, 'Global Cache' for fastest if you have enough VRAM"}),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_EXTRA_OPTIONS", {"tooltip": "Additional options to customize the caption generation"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝JoyCaption"

    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None

    def generate(self, image, model, processing_mode, prompt_style, caption_length, memory_management, extra_options=None):
        try:
            print(f"[JoyCaption GGUF] Processing image with {model} ({processing_mode} mode)")
            cache_key = f"{model}_{processing_mode}"
            cache_enabled = (memory_management == "Global Cache")
            use_subprocess = (memory_management == "Clear After Run (Subprocess)")

            # Check if we need to reload the model
            need_reload = (
                self.predictor is None or
                self.current_processing_mode != processing_mode or
                self.current_model != model
            )

            if need_reload:
                # Clean up existing model before loading new one
                if self.predictor is not None:
                    self.predictor.cleanup()
                    self.predictor = None

                # Check global cache first if caching is enabled
                if cache_enabled and cache_key in _MODEL_CACHE:
                    self.predictor = _MODEL_CACHE[cache_key]
                    print(f"[JoyCaption GGUF] Using cached model: {cache_key}")
                else:
                    try:
                        model_name = GGUF_MODELS[model]["name"]
                        # Use subprocess version for guaranteed memory release
                        if use_subprocess:
                            self.predictor = JC_GGUF_Models_Subprocess(model_name, processing_mode)
                        else:
                            self.predictor = JC_GGUF_Models(model_name, processing_mode)
                        if cache_enabled:
                            _MODEL_CACHE[cache_key] = self.predictor
                            print(f"[JoyCaption GGUF] Model cached: {cache_key}")
                    except Exception as e:
                        return (f"Error loading model: {e}",)

                self.current_processing_mode = processing_mode
                self.current_model = model

            prompt_text = build_prompt(prompt_style, caption_length, extra_options[0] if extra_options else [], extra_options[1] if extra_options else "{NAME}")

            print("[JoyCaption GGUF] Generating caption...")
            with torch.inference_mode():
                pil_image = ToPILImage()(image[0].permute(2, 0, 1))
                response = self.predictor.generate(
                    image=pil_image,
                    system=MODEL_SETTINGS["default_system_prompt"],
                    prompt=prompt_text,
                    max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
                    temperature=MODEL_SETTINGS["default_temperature"],
                    top_p=MODEL_SETTINGS["default_top_p"],
                    top_k=MODEL_SETTINGS["default_top_k"],
                )
            print("[JoyCaption GGUF] Caption generation completed")

            # Clean up after run if requested (both subprocess and regular modes)
            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)"):
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None
                self.current_model = None

            return (response,)
        except Exception as e:
            # Always clean up on error if Clear After Run is set
            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)") and self.predictor is not None:
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None
                self.current_model = None
            raise e

class JC_GGUF_adv:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always return NaN to force re-execution (no caching)
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto", "tooltip": "Auto: Automatically detect best mode\nGPU: Faster but requires more VRAM\nCPU: Slower but saves VRAM"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive", "tooltip": "Select the style of caption you want to generate"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any", "tooltip": "Control the length of the generated caption"}),
                "max_new_tokens": ("INT", {"default": MODEL_SETTINGS["default_max_tokens"], "min": 1, "max": 2048, "tooltip": "Maximum number of tokens to generate. Higher values allow longer captions"}),
                "temperature": ("FLOAT", {"default": MODEL_SETTINGS["default_temperature"], "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Control the randomness of the output. Higher values make the output more creative but less predictable"}),
                "top_p": ("FLOAT", {"default": MODEL_SETTINGS["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Control the diversity of the output. Higher values allow more diverse word choices"}),
                "top_k": ("INT", {"default": MODEL_SETTINGS["default_top_k"], "min": 0, "max": 100, "tooltip": "Limit the number of possible next tokens. Lower values make the output more focused"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom prompt template. If empty, will use the selected prompt style"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Clear After Run (Subprocess)", "Global Cache"], {"default": "Keep in Memory", "tooltip": "Choose how to manage model memory. 'Keep in Memory' for faster processing, 'Clear After Run' may leak ~850MB, 'Clear After Run (Subprocess)' guarantees 100% memory release, 'Global Cache' for fastest if you have enough VRAM"}),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_EXTRA_OPTIONS", {"tooltip": "Additional options to customize the caption generation"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝JoyCaption"

    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None

    def generate(self, image, model, processing_mode, prompt_style, caption_length, max_new_tokens, temperature, top_p, top_k, custom_prompt, memory_management, extra_options=None):
        try:
            cache_key = f"{model}_{processing_mode}"
            cache_enabled = (memory_management == "Global Cache")
            use_subprocess = (memory_management == "Clear After Run (Subprocess)")

            # Check if we need to reload the model
            need_reload = (
                self.predictor is None or
                self.current_processing_mode != processing_mode or
                self.current_model != model
            )

            if need_reload:
                # Clean up existing model before loading new one
                if self.predictor is not None:
                    self.predictor.cleanup()
                    self.predictor = None

                # Check global cache first if caching is enabled
                if cache_enabled and cache_key in _MODEL_CACHE:
                    self.predictor = _MODEL_CACHE[cache_key]
                    print(f"[JoyCaption GGUF] Using cached model: {cache_key}")
                else:
                    try:
                        model_name = GGUF_MODELS[model]["name"]
                        # Use subprocess version for guaranteed memory release
                        if use_subprocess:
                            self.predictor = JC_GGUF_Models_Subprocess(model_name, processing_mode)
                        else:
                            self.predictor = JC_GGUF_Models(model_name, processing_mode)
                        if cache_enabled:
                            _MODEL_CACHE[cache_key] = self.predictor
                            print(f"[JoyCaption GGUF] Model cached: {cache_key}")
                    except Exception as e:
                        return ("", f"Error loading model: {e}")

                self.current_processing_mode = processing_mode
                self.current_model = model

            prompt = custom_prompt if custom_prompt.strip() else build_prompt(prompt_style, caption_length, extra_options[0] if extra_options else [], extra_options[1] if extra_options else "{NAME}")
            system_prompt = MODEL_SETTINGS["default_system_prompt"]

            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            response = self.predictor.generate(
                image=pil_image,
                system=system_prompt,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Clean up after run if requested (both subprocess and regular modes)
            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)"):
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None
                self.current_model = None

            return (prompt, response)
        except Exception as e:
            # Always clean up on error if Clear After Run is set
            if memory_management in ("Clear After Run", "Clear After Run (Subprocess)") and self.predictor is not None:
                self.predictor.cleanup()
                self.predictor = None
                self.current_processing_mode = None
                self.current_model = None
            return ("", f"Error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "JC_GGUF": JC_GGUF,
    "JC_GGUF_adv": JC_GGUF_adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC_GGUF": "JoyCaption GGUF",
    "JC_GGUF_adv": "JoyCaption GGUF (Advanced)",
}
