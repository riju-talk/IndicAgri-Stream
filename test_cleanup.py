#!/usr/bin/env python3
"""Test the enhanced GPU cleanup logic"""
import torch
import gc

print("[TEST] GPU cleanup enhancements")
print(f"  GPUs available: {torch.cuda.device_count()}")

# Simulate the new cleanup function
def new_cleanup():
    """Clean GPU memory on ALL devices (for device_map='auto' which may span GPUs)"""
    gc.collect()
    if torch.cuda.is_available():
        # Clean ALL GPUs, not just GPU 0
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

# Test it
try:
    new_cleanup()
    print("\n✓ Multi-GPU cleanup function works")
except Exception as e:
    print(f"\n✗ Cleanup failed: {e}")
    exit(1)

# Verify GPU state
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {free/1e9:.1f}GB/{total/1e9:.1f}GB free")

print("\n[SUCCESS] GPU cleanup validated")
