import time
import numpy as np
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def run_latency_benchmark(n_requests=100):
    latencies = []
    payload = {"features": [0.5, -1.2, 0.3, 0.8, -0.5, 1.1, 0.0, 0.2, -0.9, 0.4]}
    for _ in range(n_requests):
        start = time.perf_counter()
        response = client.post("/predict", json=payload)
        latencies.append((time.perf_counter() - start) * 1000)
    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    print(f"Latency Benchmark Results ({n_requests} requests)")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"p95 Latency:     {p95_lat:.2f} ms")

if __name__ == "__main__":
    run_latency_benchmark()
