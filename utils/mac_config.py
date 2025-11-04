import torch
import os
import multiprocessing as mp
from contextlib import contextmanager
import threading
import warnings

class MacTorchConfig:
    """Configuration handler for PyTorch on Mac with mutex handling"""

    def __init__(self):
        self.setup_environment()
        self._lock = threading.Lock()

    def setup_environment(self):
        """Set up environment variables for Mac"""
        # Disable parallelism to avoid mutex issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Set number of threads
        if torch.backends.mps.is_available():
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Limit CPU threads to avoid mutex issues
        num_cores = mp.cpu_count()
        torch.set_num_threads(min(4, num_cores))

    def get_device(self):
        """Get appropriate device for Mac"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @contextmanager
    def mutex_guard(self):
        """Context manager for handling mutex locks"""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def optimize_model_for_mac(self, model):
        """Optimize model for Mac hardware"""
        if torch.backends.mps.is_available():
            model = model.to("mps")
            # Enable memory efficient attention if available
            if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                try:
                    model.enable_xformers_memory_efficient_attention()
                except:
                    pass
        else:
            model = model.to("cpu")
            # Use CPU optimizations
            model = torch.jit.optimize_for_inference(model)

        return model

    @staticmethod
    def handle_data_parallel(func):
        """Decorator to handle data parallel operations safely on Mac"""
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "mutex" in str(e).lower():
                        # Retry with single thread
                        torch.set_num_threads(1)
                        result = func(*args, **kwargs)
                        torch.set_num_threads(4)
                        return result
                    raise
        return wrapper

# Initialize Mac configuration
mac_config = MacTorchConfig()