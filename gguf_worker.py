"""
GGUF Worker Process - runs model in isolated process for guaranteed memory cleanup.
When process terminates, OS releases ALL its memory including CUDA.
"""
import multiprocessing as mp
import io
import base64
import gc
import os
import sys
from pathlib import Path

# Suppress output during import
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Get module directory for subprocess path setup
_MODULE_DIR = str(Path(__file__).parent.resolve())


def _worker_main(cmd_queue: mp.Queue, result_queue: mp.Queue, model_path: str, mmproj_path: str,
                 n_ctx: int, n_batch: int, n_threads: int, n_gpu_layers: int, module_dir: str):
    """
    Main worker function that runs in isolated process.
    Loads model once, processes requests until told to stop.
    """
    # Ensure module directory is in path for imports
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    import torch
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    from PIL import Image

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model = None
    chat_handler = None

    try:
        # Load model
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            chat_handler=chat_handler,
            offload_kqv=True,
            numa=True
        )

        # Signal ready
        result_queue.put({"status": "ready"})

        # Process loop
        while True:
            try:
                cmd = cmd_queue.get(timeout=1.0)
            except:
                continue

            if cmd is None or cmd.get("action") == "stop":
                break

            if cmd.get("action") == "generate":
                try:
                    # Decode image
                    img_bytes = base64.b64decode(cmd["image_b64"])
                    image = Image.open(io.BytesIO(img_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Resize
                    image = image.resize(cmd.get("image_size", (384, 384)), Image.Resampling.BILINEAR)

                    # Encode for llava
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                    data_uri = f"data:image/png;base64,{img_base64}"

                    messages = [
                        {"role": "system", "content": cmd["system"].strip()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": cmd["prompt"].strip()},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]
                        }
                    ]

                    completion_params = {
                        "messages": messages,
                        "max_tokens": cmd.get("max_new_tokens", 300),
                        "temperature": cmd.get("temperature", 0.5),
                        "top_p": cmd.get("top_p", 0.9),
                        "stop": ["</s>", "User:", "Assistant:", "USER:", "ASSISTANT:",
                                "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:",
                                "ASISTANT\n", "ASISTANT:", "ASSENT", "ASSENTED"],
                        "stream": False,
                        "repeat_penalty": 1.1,
                        "mirostat_mode": 0
                    }

                    if cmd.get("top_k", 0) > 0:
                        completion_params["top_k"] = cmd["top_k"]

                    response = model.create_chat_completion(**completion_params)
                    result = response["choices"][0]["message"]["content"].strip()

                    result_queue.put({"status": "success", "result": result})

                    # Cleanup intermediate
                    del messages, response, img_buffer
                    gc.collect()

                except Exception as e:
                    result_queue.put({"status": "error", "error": str(e)})

            elif cmd.get("action") == "ping":
                result_queue.put({"status": "pong"})

    except Exception as e:
        result_queue.put({"status": "init_error", "error": str(e)})

    finally:
        # Cleanup (though process termination will clean everything anyway)
        if model is not None:
            try:
                model.close()
            except:
                pass
        gc.collect()


class GGUFWorkerProcess:
    """
    Manages GGUF model in isolated subprocess.
    Guarantees 100% memory release when cleanup() is called by terminating the process.
    """

    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int, n_batch: int,
                 n_threads: int, n_gpu_layers: int, timeout: float = 120.0):
        self.timeout = timeout
        self._process = None
        self._cmd_queue = None
        self._result_queue = None

        # Use spawn to ensure clean CUDA state
        ctx = mp.get_context('spawn')
        self._cmd_queue = ctx.Queue()
        self._result_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_worker_main,
            args=(self._cmd_queue, self._result_queue, model_path, mmproj_path,
                  n_ctx, n_batch, n_threads, n_gpu_layers, _MODULE_DIR),
            daemon=True
        )
        self._process.start()

        # Wait for ready signal
        try:
            result = self._result_queue.get(timeout=timeout)
            if result.get("status") == "init_error":
                raise RuntimeError(f"Worker init failed: {result.get('error')}")
            if result.get("status") != "ready":
                raise RuntimeError(f"Unexpected worker status: {result}")
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Worker failed to start: {e}")

    def generate(self, image_b64: str, system: str, prompt: str,
                 max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                 image_size: tuple = (384, 384)) -> str:
        """Send generation request to worker process."""
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("Worker process is not running")

        self._cmd_queue.put({
            "action": "generate",
            "image_b64": image_b64,
            "system": system,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "image_size": image_size
        })

        try:
            result = self._result_queue.get(timeout=self.timeout)
            if result.get("status") == "error":
                return f"Generation error: {result.get('error')}"
            return result.get("result", "")
        except Exception as e:
            return f"Worker communication error: {e}"

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def cleanup(self):
        """
        Terminate worker process - guarantees 100% memory release.
        OS will free ALL process memory including leaked CUDA memory.
        """
        had_process = self._process is not None

        if self._process is not None:
            try:
                # Try graceful shutdown first
                if self._process.is_alive() and self._cmd_queue is not None:
                    try:
                        self._cmd_queue.put({"action": "stop"}, timeout=1.0)
                    except:
                        pass
                    self._process.join(timeout=2.0)

                # Force kill if still alive
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)

                # Last resort
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
            except:
                pass
            finally:
                self._process = None

        # Close queues
        for q in [self._cmd_queue, self._result_queue]:
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except:
                    pass

        self._cmd_queue = None
        self._result_queue = None

        gc.collect()
        if had_process:
            print("[JoyCaption GGUF] Worker process terminated - memory fully released")
