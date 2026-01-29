"""
GGUF Worker Process - runs model in isolated process for guaranteed memory cleanup.
When process terminates, OS releases ALL its memory including CUDA.

This module can be run as a standalone script for subprocess isolation.
"""
import sys
import os
import io
import json
import base64
import gc
import time
from pathlib import Path

_last_progress_time = 0
_progress_throttle_interval = 5.0

# Suppress output during import
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def run_worker(config: dict):
    """
    Main worker function that runs in isolated process.
    Loads model once, processes requests until stdin closes.

    Communication protocol (JSON lines over stdin/stdout):
    - Input: {"action": "generate", ...} or {"action": "stop"}
    - Output: {"status": "ready"}, {"status": "success", "result": "..."}, {"status": "error", "error": "..."}
    """
    import torch
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    from PIL import Image

    def send_response(data: dict):
        """Send JSON response to parent process."""
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    def log(msg: str):
        """Log to stderr (doesn't interfere with JSON protocol on stdout)."""
        print(f"[GGUF Worker] {msg}", file=sys.stderr, flush=True)

    def send_progress(message: str):
        """Send progress update to parent process."""
        send_response({"status": "progress", "message": message})

    log("Process started")
    send_progress("Worker process started")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model = None
    chat_handler = None

    try:
        model_path = config["model_path"]
        mmproj_path = config["mmproj_path"]
        n_ctx = config["n_ctx"]
        n_batch = config["n_batch"]
        n_threads = config["n_threads"]
        n_gpu_layers = config["n_gpu_layers"]

        log(f"Loading CLIP model from {mmproj_path}...")
        send_progress("Loading CLIP vision model...")
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        send_progress("CLIP model loaded")

        log(f"Loading LLM from {model_path}...")
        send_progress("Loading language model into memory...")
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
        log("Models loaded successfully")
        send_progress("Language model loaded successfully")

        # Signal ready
        send_response({"status": "ready"})

        # Process loop - read JSON commands from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as e:
                send_response({"status": "error", "error": f"Invalid JSON: {e}"})
                continue

            if cmd.get("action") == "stop":
                log("Received stop command")
                break

            if cmd.get("action") == "generate":
                try:
                    send_progress("Processing image in worker...")
                    # Decode image
                    img_bytes = base64.b64decode(cmd["image_b64"])
                    image = Image.open(io.BytesIO(img_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Resize
                    image_size = tuple(cmd.get("image_size", [384, 384]))
                    image = image.resize(image_size, Image.Resampling.BILINEAR)

                    send_progress("Encoding image for model...")
                    # Encode for llava
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                    data_uri = f"data:image/png;base64,{img_base64}"

                    send_progress("Preparing prompt...")
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

                    import random
                    completion_params = {
                        "messages": messages,
                        "max_tokens": cmd.get("max_new_tokens", 300),
                        "temperature": cmd.get("temperature", 0.5),
                        "top_p": cmd.get("top_p", 0.9),
                        "seed": random.randint(1, 2**31 - 1),  # Random seed for variety
                        "stop": ["</s>", "User:", "Assistant:", "USER:", "ASSISTANT:",
                                "\nUser:", "\nAssistant:", "\nUSER:", "\nASSISTANT:",
                                "ASISTANT\n", "ASISTANT:", "ASSENT", "ASSENTED"],
                        "stream": False,
                        "repeat_penalty": 1.1,
                        "mirostat_mode": 0
                    }

                    if cmd.get("top_k", 0) > 0:
                        completion_params["top_k"] = cmd["top_k"]

                    send_progress("Generating caption with model...")
                    response = model.create_chat_completion(**completion_params)
                    result = response["choices"][0]["message"]["content"].strip()

                    log(f"Generation complete, result length: {len(result)}")
                    send_progress("Caption generation complete")
                    log("Sending success response...")
                    send_response({"status": "success", "result": result})
                    log("Success response sent")

                    # Cleanup intermediate
                    del messages, response, img_buffer
                    gc.collect()

                except Exception as e:
                    log(f"Generation error: {e}")
                    send_response({"status": "error", "error": str(e)})

            elif cmd.get("action") == "ping":
                send_response({"status": "pong"})

    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_response({"status": "init_error", "error": str(e)})

    finally:
        log("Shutting down...")
        if model is not None:
            try:
                model.close()
            except:
                pass
        gc.collect()
        log("Process exit")


class GGUFWorkerProcess:
    """
    Manages GGUF model in isolated subprocess using Popen.
    Guarantees 100% memory release when cleanup() is called by terminating the process.
    """

    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int, n_batch: int,
                 n_threads: int, n_gpu_layers: int, timeout: float = 180.0):
        import subprocess
        import threading
        import queue

        self.timeout = timeout
        self._process = None
        self._stdout_queue = queue.Queue()
        self._reader_thread = None

        # Config to pass to worker
        config = {
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers
        }

        # Get path to this module
        worker_script = Path(__file__).resolve()

        # Start subprocess with unbuffered stdout
        print(f"[JoyCaption GGUF] Starting worker subprocess...")
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(worker_script), json.dumps(config)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Start stdout reader thread
        def read_stdout():
            try:
                for line in self._process.stdout:
                    self._stdout_queue.put(line)
                print("[JoyCaption GGUF] Reader thread: stdout EOF reached")
            except Exception as e:
                print(f"[JoyCaption GGUF] Reader thread: exception: {e}")
            self._stdout_queue.put(None)  # Signal EOF

        self._reader_thread = threading.Thread(target=read_stdout, daemon=True)
        self._reader_thread.start()

        # Start stderr reader thread
        def read_stderr():
            try:
                for line in self._process.stderr:
                    print(f"[JoyCaption Worker STDERR] {line.rstrip()}")
            except:
                pass

        threading.Thread(target=read_stderr, daemon=True).start()

        # Wait for ready signal
        try:
            print(f"[JoyCaption GGUF] Waiting for worker ready signal...")
            while True:
                response = self._read_response(timeout=timeout)
                if response is None:
                    raise RuntimeError("No response from worker")

                status = response.get("status")

                if status == "progress":
                    global _last_progress_time
                    current_time = time.time()
                    if current_time - _last_progress_time >= _progress_throttle_interval:
                        try:
                            from server import PromptServer
                            if hasattr(PromptServer, 'instance') and PromptServer.instance:
                                PromptServer.instance.send_sync("progress", {
                                    "message": response.get("message", "Initializing worker...")
                                })
                                _last_progress_time = current_time
                                print(f"[JoyCaption GGUF] Worker init: {response.get('message')}")
                        except Exception as e:
                            print(f"[JoyCaption GGUF] Failed to forward init progress: {e}")
                    continue

                if status == "init_error":
                    raise RuntimeError(f"Worker init failed: {response.get('error')}")
                elif status == "ready":
                    print(f"[JoyCaption GGUF] Worker ready")
                    break
                else:
                    continue

        except Exception as e:
            print(f"[JoyCaption GGUF] Worker startup error: {e}")
            self.cleanup()
            raise RuntimeError(f"Worker failed to start: {e}")

    def _read_response(self, timeout: float = None) -> dict:
        """Read JSON response from worker stdout via queue."""
        import queue as queue_module
        import time

        timeout = timeout or self.timeout
        start_time = time.time()

        while True:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                print(f"[JoyCaption GGUF] _read_response: total timeout after {timeout}s")
                return None

            try:
                line = self._stdout_queue.get(timeout=remaining)
                if line is None:
                    print("[JoyCaption GGUF] _read_response: got EOF (None)")
                    return None

                stripped = line.strip()
                if not stripped:
                    continue

                # Try to parse as JSON
                try:
                    result = json.loads(stripped)
                    print(f"[JoyCaption GGUF] _read_response: parsed JSON with status={result.get('status')}")
                    return result
                except json.JSONDecodeError:
                    # Skip non-JSON output from llama_cpp (like "encoding image slice...")
                    print(f"[JoyCaption GGUF] _read_response: skipping non-JSON: {stripped[:100]}")
                    continue

            except queue_module.Empty:
                print(f"[JoyCaption GGUF] _read_response: queue timeout")
                return None

    def _send_command(self, cmd: dict):
        """Send JSON command to worker stdin."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Worker process is not running")
        self._process.stdin.write(json.dumps(cmd) + "\n")
        self._process.stdin.flush()

    def generate(self, image_b64: str, system: str, prompt: str,
                 max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                 image_size: tuple = (384, 384)) -> str:
        """Send generation request to worker process."""
        if not self.is_alive():
            raise RuntimeError("Worker process is not running")

        self._send_command({
            "action": "generate",
            "image_b64": image_b64,
            "system": system,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "image_size": list(image_size)
        })

        # Read responses until we get success or error
        while True:
            response = self._read_response(timeout=self.timeout)
            if response is None:
                return "Error: No response from worker"

            status = response.get("status")

            if status == "progress":
                global _last_progress_time
                current_time = time.time()
                if current_time - _last_progress_time >= _progress_throttle_interval:
                    try:
                        from server import PromptServer
                        if hasattr(PromptServer, 'instance') and PromptServer.instance:
                            PromptServer.instance.send_sync("progress", {
                                "message": response.get("message", "Worker processing...")
                            })
                            _last_progress_time = current_time
                            print(f"[JoyCaption GGUF] Worker: {response.get('message')}")
                    except Exception as e:
                        print(f"[JoyCaption GGUF] Failed to forward progress: {e}")
                continue

            if status == "error":
                return f"Generation error: {response.get('error')}"
            elif status == "success":
                return response.get("result", "")
            else:
                continue

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def cleanup(self):
        """
        Terminate worker process - guarantees 100% memory release.
        OS will free ALL process memory including leaked CUDA memory.
        """
        if self._process is None:
            return

        try:
            if self.is_alive():
                try:
                    self._send_command({"action": "stop"})
                    self._process.wait(timeout=2.0)
                except:
                    pass

            if self.is_alive():
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except:
                    pass

            if self.is_alive():
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                except:
                    pass

        except:
            pass
        finally:
            for pipe in [self._process.stdin, self._process.stdout, self._process.stderr]:
                if pipe:
                    try:
                        pipe.close()
                    except:
                        pass
            self._process = None

        gc.collect()
        print("[JoyCaption GGUF] Worker process terminated - memory fully released")


# Entry point when run as subprocess
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gguf_worker.py <config_json>", file=sys.stderr)
        sys.exit(1)

    try:
        config = json.loads(sys.argv[1])
        run_worker(config)
    except Exception as e:
        print(json.dumps({"status": "init_error", "error": str(e)}))
        sys.exit(1)
