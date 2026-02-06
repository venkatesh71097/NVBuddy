import json
import time
import random

def _simulate_triton_inference(model_name: str, batch_size: int, precision: str = "FP16"):
    """
    Simulates a Triton Inference Server request.
    Returns simulated latency and throughput metrics.
    """
    base_latency_ms = 20  # Overhead
    
    # Model complexity factors
    if "llama" in model_name.lower() or "gpt" in model_name.lower():
        compute_per_token = 50 
    else:
        compute_per_token = 10
        
    # Precision factors
    if precision == "FP8":
        prec_factor = 0.5
    elif precision == "INT8":
        prec_factor = 0.6
    else: # FP16
        prec_factor = 1.0
        
    # Latency Calculation: (Batch * Compute * Precision) + Overhead
    # Simulates that larger batches take longer to process, but have higher throughput
    compute_time = (batch_size * compute_per_token * prec_factor) / 10 # Scale down for ms
    total_latency_ms = base_latency_ms + compute_time
    
    # Throughput: Requests per Second
    throughput_rps = 1000 / total_latency_ms * batch_size
    
    results = {
        "status": "SUCCESS",
        "server_state": "ONLINE",
        "model": model_name,
        "metrics": {
            "batch_size": batch_size,
            "precision": precision,
            "p99_latency_ms": round(total_latency_ms, 2),
            "throughput_req_per_sec": round(throughput_rps, 2)
        }
    }
    
    # Educational explanation attached to the tool output
    explanation = f"""
    TRITON SIMULATION ANALYTICS:
    ----------------------------
    Running '{model_name}' at Batch Size {batch_size}.
    
    1. **Latency Impact**: Processing {batch_size} items took {results['metrics']['p99_latency_ms']}ms.
       * Notice that latency INCREASES with batch size.
       
    2. **Throughput Impact**: We processed {results['metrics']['throughput_req_per_sec']} requests/second.
       * Notice that throughput INCREASES because we parallelize the math.
       
    3. **Trade-off**: For real-time apps (chatbots), keep Batch Size low (<4) for low latency.
       For offline processors (summarization), max out Batch Size for high throughput.
    """
    
    return json.dumps(results, indent=2) + "\n" + explanation
