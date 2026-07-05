import requests
import os

MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:5000")

def allocate_container():
    """Allocate a container and return its ID"""
    resp = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": "python"})
    assert resp.status_code == 200
    return resp.json()["container_id"]

def test_continuous_execution():
    container_id = allocate_container()
    
    # Declare variable
    resp1 = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "x = 42"})
    assert resp1.status_code == 200
    
    # Use variable from previous execution
    resp2 = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "print(x)"})
    assert resp2.status_code == 200
    outputs = resp2.json()["outputs"]
    assert any("42" in str(output) for output in outputs)
    print("✓ Continuous execution test passed")
    
    # Clean up
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")

def test_timeout():
    container_id = allocate_container()
    
    resp = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "import time; time.sleep(2)"})
    assert resp.status_code == 200
    print("✓ Timeout test passed")
    
    # Clean up
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")

def test_exception():
    container_id = allocate_container()
    
    resp = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "1/0"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert any(output.get("type") == "error" for output in outputs)
    print("✓ Exception test passed")
    
    # Clean up
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")

def test_container_health():
    container_id = allocate_container()
    
    resp = requests.get(f"{MANAGER_URL}/session/{container_id}/health")
    assert resp.status_code == 200
    health_data = resp.json()
    assert health_data["status"] == "ok"
    print("✓ Container health test passed")
    
    # Clean up
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")

if __name__ == "__main__":
    test_continuous_execution()
    test_timeout()
    test_exception()
    test_container_health()
    print("All tests passed!") 