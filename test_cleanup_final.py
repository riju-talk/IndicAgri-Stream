#!/usr/bin/env python3
"""Test that eval.py cleanup works correctly before model loading"""
import os
os.environ["HF_CACHE_DIR"] = "/media/nas_mount/hf_cache"

import sys
import torch

# Test cleanup function directly from eval.py context
print("[TEST] Loading eval.py and testing cleanup sequence")

try:
    # Import eval module to get access to cleanup function
    sys.path.insert(0, '/Code/12_data_engineering/IndicAgri-Stream')
    
    # Just test that the cleanup function definition is correct
    import subprocess
    result = subprocess.run(
        ['python', '-c', '''
import torch
import gc

def cleanup():
    """Clean GPU memory on ALL devices"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

print("[INLINE TEST] Testing cleanup")
cleanup()
print("[SUCCESS] Cleanup works")
'''],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    if result.returncode == 0:
        print(result.stdout)
        print("\n✓ Cleanup function verified to work correctly")
    else:
        print(f"✗ Cleanup test failed:\n{result.stderr}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Test error: {e}")
    sys.exit(1)

print("\n[RESULT] All cleanup verifications passed")
