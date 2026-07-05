#!/usr/bin/env python3
"""
Container lifecycle test: variable persistence and restart functionality
"""

import httpx
import asyncio
import time
import random

MANAGER_URL = "http://localhost:5000"

async def wait_for_restart_complete(client: httpx.AsyncClient, container_id: int, max_wait: int = 30) -> bool:
    """Wait for container restart to complete and kernel to be ready"""
    
    # First, wait for restart to complete
    for _ in range(max_wait):
        try:
            response = await client.get(f"{MANAGER_URL}/session/{container_id}/restart-status")
            if response.status_code == 200:
                status_data = response.json()
                if not status_data.get("restarting", False):
                    break  # Restart completed
        except:
            pass
        await asyncio.sleep(1)
    else:
        return False  # Timeout waiting for restart to complete
    
    # Then, wait for kernel to be ready
    for _ in range(max_wait):
        try:
            response = await client.get(f"{MANAGER_URL}/session/{container_id}/ready")
            if response.status_code == 200:
                ready_data = response.json()
                if ready_data.get("ready", False):
                    return True  # Kernel is ready
        except:
            pass
        await asyncio.sleep(1)
    
    return False  # Timeout waiting for kernel to be ready

async def run_single_iteration(client: httpx.AsyncClient, container_id: int, iteration: int) -> dict:
    """Run one complete iteration for a container"""
    # Randomly choose x value (5 or 8)
    x_value = random.choice([5, 8])
    
    iteration_result = {
        'iteration': iteration,
        'x_value': x_value,
        'operations': {
            'set_variable': {'success': False, 'error': None},
            'print_variable': {'success': False, 'error': None, 'expected': str(x_value)},
            'restart_container': {'success': False, 'error': None},
            'print_after_restart': {'success': False, 'error': None, 'expected_error': True}
        }
    }
    
    try:
        # Operation 1: Set x variable
        response = await client.post(
            f"{MANAGER_URL}/session/{container_id}/execute",
            json={"code": f"x = {x_value}"}
        )
        if response.status_code == 200:
            iteration_result['operations']['set_variable']['success'] = True
        else:
            iteration_result['operations']['set_variable']['error'] = f"HTTP {response.status_code}"
            
        # Operation 2: Print x and verify
        response = await client.post(
            f"{MANAGER_URL}/session/{container_id}/execute",
            json={"code": "print(x)"}
        )
        if response.status_code == 200:
            outputs = response.json().get("outputs", [])
            if any(str(x_value) in str(output) for output in outputs):
                iteration_result['operations']['print_variable']['success'] = True
            else:
                iteration_result['operations']['print_variable']['error'] = f"Expected {x_value}, got {outputs}"
        else:
            iteration_result['operations']['print_variable']['error'] = f"HTTP {response.status_code}"
            
        # Operation 3: Restart container
        response = await client.post(f"{MANAGER_URL}/session/{container_id}/restart")
        if response.status_code == 200:
            # Wait for restart to complete
            restart_success = await wait_for_restart_complete(client, container_id)
            if restart_success:
                # Add longer stabilization delay to ensure kernel is fully ready
                await asyncio.sleep(5)
                iteration_result['operations']['restart_container']['success'] = True
            else:
                iteration_result['operations']['restart_container']['error'] = "Restart timeout or failed"
        else:
            iteration_result['operations']['restart_container']['error'] = f"HTTP {response.status_code}"
            
        # Operation 4: Print x after restart (should fail)
        response = await client.post(
            f"{MANAGER_URL}/session/{container_id}/execute",
            json={"code": "print(x)"}
        )
        if response.status_code == 200:
            outputs = response.json().get("outputs", [])
            # Check if any output contains error (NameError, etc.)
            has_error = any(
                output.get("type") == "error" and "NameError" in str(output) 
                for output in outputs
            )
            if has_error:
                iteration_result['operations']['print_after_restart']['success'] = True
            else:
                iteration_result['operations']['print_after_restart']['error'] = f"Expected NameError, got {outputs}"
        else:
            iteration_result['operations']['print_after_restart']['error'] = f"HTTP {response.status_code}"
            
    except Exception as e:
        iteration_result['operations']['set_variable']['error'] = f"Exception: {e}"
    
    return iteration_result

async def test_container_lifecycle(container_id: int) -> dict:
    """Test one container through 3 complete cycles"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = {
            'container_id': container_id,
            'iterations': [],
            'total_operations': 0,
            'successful_operations': 0
        }
        
        for iteration in range(3):
            iteration_result = await run_single_iteration(client, container_id, iteration)
            results['iterations'].append(iteration_result)
            
            # Count operations
            for op_name, op_result in iteration_result['operations'].items():
                results['total_operations'] += 1
                if op_result['success']:
                    results['successful_operations'] += 1
        
        return results

async def run_all_containers():
    """Run test on all 100 containers concurrently"""
    print("🚀 Container Lifecycle Test")
    print("="*50)
    print("Testing variable persistence and restart functionality...")
    print("Each container: 3 iterations × 4 operations = 12 operations")
    print("Total operations: 100 containers × 12 = 1,200 operations")
    print()
    
    start_time = time.time()
    
    tasks = []
    for container_id in range(1, 101):
        task = test_container_lifecycle(container_id)
        tasks.append(task)
    
    print("⏳ Running all containers concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Process results
    total_containers = 0
    successful_containers = 0
    total_operations = 0
    successful_operations = 0
    failed_containers = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_containers.append(f"Container exception: {result}")
            continue
        
        # Only process non-exception results
        if not isinstance(result, dict):
            failed_containers.append(f"Invalid result type: {type(result)}")
            continue
            
        total_containers += 1
        total_operations += result['total_operations']
        successful_operations += result['successful_operations']
        
        if result['successful_operations'] == result['total_operations']:
            successful_containers += 1
        else:
            failed_containers.append(f"Container {result['container_id']}: {result['successful_operations']}/{result['total_operations']} operations")
    
    # Report results
    print(f"⏱️  Total time: {duration:.2f}s")
    print("📊 Results:")
    print(f"   ✅ Successful containers: {successful_containers}/{total_containers}")
    print(f"   ✅ Successful operations: {successful_operations}/{total_operations}")
    print(f"   📈 Success rate: {successful_operations/total_operations*100:.1f}%")
    print(f"   ⚡ Operations per second: {total_operations/duration:.1f}")
    
    if failed_containers:
        print("\n❌ Issues found:")
        for failure in failed_containers[:10]:  # Show first 10 failures
            print(f"   {failure}")
        if len(failed_containers) > 10:
            print(f"   ... and {len(failed_containers) - 10} more")
    
    print("\n" + "="*50)
    print("📊 Lifecycle Test Complete")

if __name__ == "__main__":
    asyncio.run(run_all_containers()) 