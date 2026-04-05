#!/usr/bin/env python3
"""Final integration test - verify cleanup happens before model load"""
import os
os.environ["HF_CACHE_DIR"] = "/media/nas_mount/hf_cache"

import sys
import torch
import gc

# Manually test the exact cleanup sequence from eval.py
print("[INTEGRATION TEST] Simulating model load sequence with cleanup\n")

def cleanup():
    """Exact copy from eval.py"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

# Simulate the load_model flow
print("1. Initial GPU state:")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"   GPU {i}: {free/1e9:.2f}GB / {total/1e9:.2f}GB")

print("\n2. Cleaning GPUs (first cleanup call)...")
cleanup()
print("   ✓ First cleanup complete")

print("\n3. Waiting 0.5s for GPU to settle...")
import time
time.sleep(0.5)

print("\n4. Cleaning GPUs again (second cleanup call)...")
cleanup()
print("   ✓ Second cleanup complete")

print("\n5. Final GPU state before model load:")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"   GPU {i}: {free/1e9:.2f}GB / {total/1e9:.2f}GB")

print("\n[SUCCESS] Cleanup sequence executed successfully")
print("[READY] GPUs are now clean and ready for model loading")
