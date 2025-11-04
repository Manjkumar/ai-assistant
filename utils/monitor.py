import psutil
import torch
import time
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = []

    def start(self):
        self.start_time = time.time()

    def log_metrics(self):
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }

        if torch.backends.mps.is_available():
            # Log MPS memory if available
            try:
                metrics["mps_memory"] = torch.mps.current_allocated_memory()
            except:
                pass

        self.metrics.append(metrics)
        return metrics

    def report(self):
        if not self.metrics:
            return "No metrics collected"

        avg_cpu = sum(m["cpu_percent"] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m["memory_percent"] for m in self.metrics) / len(self.metrics)
        total_time = self.metrics[-1]["elapsed_time"] if self.metrics else 0

        return f"""
        Performance Report:
        - Average CPU Usage: {avg_cpu:.2f}%
        - Average Memory Usage: {avg_memory:.2f}%
        - Total Execution Time: {total_time:.2f} seconds
        - Samples Processed: {len(self.metrics)}
        """