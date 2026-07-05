import requests

INSTANCE_URL = "http://localhost:60000"

def test_execute_code():
    resp = requests.post(f"{INSTANCE_URL}/execute", json={"code": "print('hello')"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert any("hello" in str(output) for output in outputs)
    print("✓ Execute code test passed")

def test_continuous_execution():
    resp1 = requests.post(f"{INSTANCE_URL}/execute", json={"code": "x = 42"})
    assert resp1.status_code == 200
    
    resp2 = requests.post(f"{INSTANCE_URL}/execute", json={"code": "print(x)"})
    assert resp2.status_code == 200
    outputs = resp2.json()["outputs"]
    assert any("42" in str(output) for output in outputs)
    print("✓ Continuous execution test passed")

def test_exception_handling():
    resp = requests.post(f"{INSTANCE_URL}/execute", json={"code": "1/0"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert any(output.get("type") == "error" for output in outputs)
    print("✓ Exception handling test passed")

def test_health_endpoint():
    resp = requests.get(f"{INSTANCE_URL}/health")
    assert resp.status_code == 200
    health_data = resp.json()
    assert health_data["status"] == "ok"
    print("✓ Health endpoint test passed")

def test_ready_endpoint():
    resp = requests.get(f"{INSTANCE_URL}/ready")
    assert resp.status_code == 200
    ready_data = resp.json()
    assert "ready" in ready_data
    print("✓ Ready endpoint test passed")

def test_restart_endpoint():
    resp = requests.post(f"{INSTANCE_URL}/restart")
    assert resp.status_code == 200
    restart_data = resp.json()
    assert restart_data["status"] in ["restarting", "already_restarting"]
    print("✓ Restart endpoint test passed")

def test_restart_status():
    resp = requests.get(f"{INSTANCE_URL}/restart-status")
    assert resp.status_code == 200
    status_data = resp.json()
    assert "restarting" in status_data
    print("✓ Restart status test passed")

def test_timeout_handling():
    resp = requests.post(f"{INSTANCE_URL}/execute", json={"code": "import time; time.sleep(5)"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert len(outputs) > 0
    assert any(output.get("type") == "error" and "timed out" in output.get("value", "").lower() for output in outputs)
    print("✓ Timeout handling test passed")

def test_complex_code_execution():
    code = """
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
"""
    resp = requests.post(f"{INSTANCE_URL}/execute", json={"code": code})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    assert any("4.0" in str(output) for output in outputs)
    print("✓ Complex code execution test passed")

if __name__ == "__main__":
    test_execute_code()
    test_continuous_execution()
    test_exception_handling()
    test_health_endpoint()
    test_ready_endpoint()
    test_restart_endpoint()
    test_restart_status()
    test_code_info_no_folder()
    test_timeout_handling()
    test_complex_code_execution()
    print("All instance tests passed!") 