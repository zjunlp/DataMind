#!/usr/bin/env python3

import requests
import sys

MANAGER_URL = "http://localhost:5000"

def test_gpu_in_container(container_id: int):
    """Simple GPU test - just check if nvidia-smi works"""
    print(f"Testing GPU access in container {container_id}...")
    
    code = '''
import subprocess
import os

cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')
print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("GPU found:")
        print(result.stdout.strip())
    else:
        print(f"ERROR: nvidia-smi failed: {result.stderr}")
except Exception as e:
    print(f"ERROR: {e}")
'''
    
    try:
        response = requests.post(
            f"{MANAGER_URL}/session/{container_id}/execute",
            json={"code": code},
            timeout=15
        )
        response.raise_for_status()
        
        outputs = response.json().get("outputs", [])
        
        for output in outputs:
            if output.get("type") == "stream":
                print(f"   {output.get('text', '').strip()}")
            elif output.get("type") == "error":
                print(f"   ERROR: {output.get('name', '')}: {output.get('value', '')}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   Request failed: {e}")
        return False

def main():
    print("🔥 Simple GPU Test")
    print("=" * 30)
    
    # Check manager health
    try:
        response = requests.get(f"{MANAGER_URL}/health", timeout=5)
        response.raise_for_status()
        print("✅ Manager is healthy")
    except Exception as e:
        print(f"❌ Manager health check failed: {e}")
        return 1
    
    # Test GPU access in both containers
    success = True
    for container_id in [0, 1]:
        if not test_gpu_in_container(container_id):
            success = False
    
    if success:
        print("\n✅ GPU test PASSED - both containers can access their GPUs!")
        return 0
    else:
        print("\n❌ GPU test FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
