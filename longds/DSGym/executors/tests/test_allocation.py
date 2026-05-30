import requests
import time
import os

MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:5000")

def test_allocation_deallocation():
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    data = response.json()
    container_id1 = data["container_id"]
    
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    data = response.json()
    container_id2 = data["container_id"]
    assert container_id1 != container_id2
    
    response = requests.post(f"{MANAGER_URL}/deallocate/{container_id1}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deallocated"
    assert data["container_id"] == container_id1
    
    time.sleep(3)
    
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    data = response.json()
    container_id3 = data["container_id"]
    
    requests.post(f"{MANAGER_URL}/deallocate/{container_id2}")
    requests.post(f"{MANAGER_URL}/deallocate/{container_id3}")

def test_allocation_flow():
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    container_id = response.json()["container_id"]
    
    response = requests.post(
        f"{MANAGER_URL}/session/{container_id}/execute",
        json={"code": "test_var = 42\n42"}
    )
    assert response.status_code == 200
    outputs = response.json()["outputs"]
    assert len(outputs) > 0
    
    response = requests.post(f"{MANAGER_URL}/deallocate/{container_id}")
    assert response.status_code == 200 

def test_status_tracking():
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    initial_status = response.json()
    
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    container_id = response.json()["container_id"]
    
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    status_after_allocation = response.json()
    
    assert status_after_allocation["allocated_containers"] == initial_status["allocated_containers"] + 1
    assert status_after_allocation["available_containers"] == initial_status["available_containers"] - 1
    assert container_id in status_after_allocation["allocated_container_ids"]
    assert status_after_allocation["total_containers"] == initial_status["total_containers"]
    
    response = requests.post(f"{MANAGER_URL}/deallocate/{container_id}")
    assert response.status_code == 200
    
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    status_after_deallocation = response.json()
    
    assert status_after_deallocation["allocated_containers"] == initial_status["allocated_containers"]
    assert status_after_deallocation["available_containers"] == initial_status["available_containers"]
    assert container_id not in status_after_deallocation["allocated_container_ids"]
    assert status_after_deallocation["total_containers"] == initial_status["total_containers"]

def test_deallocate_unallocated_container():
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    status = response.json()
    
    non_allocated_container_id = 999
    if non_allocated_container_id in status["allocated_container_ids"]:
        non_allocated_container_id = 1000
    
    response = requests.post(f"{MANAGER_URL}/deallocate/{non_allocated_container_id}")
    assert response.status_code == 400
    assert "not allocated" in response.json()["detail"] 