import requests
import os
import re

MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:5000")
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

def allocate_container():
    resp = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": "python"})
    assert resp.status_code == 200
    return resp.json()["container_id"]

def test_stream_output_exact():
    container_id = allocate_container()
    
    resp = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "print('Hello World')"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    
    expected = [
        {
            "type": "stream",
            "name": "stdout",
            "text": "Hello World\n"
        }
    ]
    
    assert outputs == expected
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")

def test_error_output_exact():
    container_id = allocate_container()
    
    resp = requests.post(f"{MANAGER_URL}/session/{container_id}/execute", json={"code": "nprint(x)"})
    assert resp.status_code == 200
    outputs = resp.json()["outputs"]
    
    assert len(outputs) == 1
    assert outputs[0]["type"] == "error"
    assert outputs[0]["name"] == "NameError"
    assert outputs[0]["value"] == "name 'nprint' is not defined"
    
    traceback = outputs[0]["traceback"]
    assert len(traceback) == 3
    assert "NameError" in traceback[0]
    assert "Traceback (most recent call last)" in traceback[0]
    assert "nprint(x)" in traceback[1]
    assert "NameError: name 'nprint' is not defined" in traceback[2]
    
    for line in traceback:
        assert ANSI_PATTERN.search(line) is None
        assert not line.strip().startswith('---')
    
    requests.post(f"{MANAGER_URL}/deallocate/{container_id}")
