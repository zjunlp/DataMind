import requests
import time
import os

MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:5000")

def test_container_type_allocation():
    """Test that containers can be allocated by specific type"""
    
    # Get initial status
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    initial_status = response.json()
    
    # Verify we have multiple types available
    assert "available_by_type" in initial_status
    available_types = initial_status["available_by_type"]
    assert len(available_types) >= 2, f"Expected at least 2 container types, got: {available_types}"
    
    print(f"Available container types: {available_types}")
    
    # Get the first two available types
    type1, type2 = list(available_types.keys())[:2]
    
    # Allocate container of first type
    response = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": type1})
    assert response.status_code == 200
    data = response.json()
    container1_id = data["container_id"]
    
    # Allocate container of second type
    response = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": type2})
    assert response.status_code == 200
    data = response.json()
    container2_id = data["container_id"]
    
    # Verify different containers were allocated
    assert container1_id != container2_id, "Same container allocated for different types"
    
    # Check status after allocation
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    after_status = response.json()
    
    # Verify allocated count increased
    assert after_status["allocated_containers"] == initial_status["allocated_containers"] + 2
    
    # Verify type-specific counts decreased
    after_available = after_status["available_by_type"]
    assert after_available[type1] == available_types[type1] - 1
    assert after_available[type2] == available_types[type2] - 1
    
    print(f"✅ Successfully allocated {type1} container: {container1_id}")
    print(f"✅ Successfully allocated {type2} container: {container2_id}")

def test_invalid_container_type():
    """Test that invalid container types are rejected"""
    
    response = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": "invalid_type"})
    assert response.status_code == 400
    assert "Unknown container type" in response.json()["detail"]
    
    print("✅ Invalid container type correctly rejected")

def test_container_type_deallocation():
    """Test that deallocated containers return to correct type queue"""
    
    # Get available types
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    initial_status = response.json()
    available_types = initial_status["available_by_type"]
    
    # Pick a type that has available containers
    container_type = next(type_name for type_name, count in available_types.items() if count > 0)
    initial_count = available_types[container_type]
    
    # Allocate container of specific type
    response = requests.post(f"{MANAGER_URL}/allocate", params={"container_type": container_type})
    assert response.status_code == 200
    container_id = response.json()["container_id"]
    
    # Verify count decreased
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    after_alloc = response.json()
    assert after_alloc["available_by_type"][container_type] == initial_count - 1
    
    # Deallocate container
    response = requests.post(f"{MANAGER_URL}/deallocate/{container_id}")
    assert response.status_code == 200
    
    # Wait for restart
    time.sleep(3)
    
    # Verify count returned to original
    response = requests.get(f"{MANAGER_URL}/status")
    assert response.status_code == 200
    after_dealloc = response.json()
    assert after_dealloc["available_by_type"][container_type] == initial_count
    
    print(f"✅ Container {container_id} correctly returned to {container_type} queue")

def test_any_type_allocation():
    """Test that allocation without specifying type works (fallback behavior)"""
    
    # Allocate without specifying type
    response = requests.post(f"{MANAGER_URL}/allocate")
    assert response.status_code == 200
    container_id = response.json()["container_id"]
    
    # Should succeed and return some container
    assert isinstance(container_id, int)
    
    # Deallocate for cleanup
    response = requests.post(f"{MANAGER_URL}/deallocate/{container_id}")
    assert response.status_code == 200
    
    print(f"✅ Any-type allocation worked: {container_id}")

if __name__ == "__main__":
    print("Running container type tests...")
    
    test_container_type_allocation()
    test_invalid_container_type()
    test_container_type_deallocation()
    test_any_type_allocation()
    
    print("\n🎉 All container type tests passed!")