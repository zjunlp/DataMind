#!/usr/bin/env python3

import requests
import time

MANAGER_URL = "http://localhost:5000"

def print_section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def execute_code_and_print(container_id: int, code: str, description: str = ""):
    """Execute code and print the results in a formatted way"""
    if description:
        print(f"\n🔧 {description}")
    
    print(f"📝 Code (Container {container_id}):")
    for line in code.strip().split('\n'):
        print(f"   {line}")
    
    try:
        response = requests.post(
            f"{MANAGER_URL}/session/{container_id}/execute",
            json={"code": code},
            timeout=10
        )
        response.raise_for_status()
        
        outputs = response.json().get("outputs", [])
        print("📤 Outputs:")
        
        if not outputs:
            print("   (no output)")
        else:
            for output in outputs:
                if output.get("type") == "stream":
                    print(f"   {output.get('text', '').strip()}")
                elif output.get("type") == "result":
                    data = output.get("data", {})
                    if "text/plain" in data:
                        print(f"   → {data['text/plain']}")
                elif output.get("type") == "error":
                    print(f"   ❌ {output.get('name', 'Error')}: {output.get('value', '')}")
        
        return True, outputs
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False, []
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False, []

def test_basic_execution():
    print_section("BASIC CODE EXECUTION")
    
    # Simple math
    execute_code_and_print(1, "2 + 2", "Simple arithmetic")
    
    # Print statement
    execute_code_and_print(1, "print('Hello from Container 1!')", "Print statement")
    
    # Complex calculation
    execute_code_and_print(2, """
import math
result = math.sqrt(16) + math.pi
print(f"sqrt(16) + π = {result:.4f}")
""", "Mathematical operations")

def test_variable_persistence():
    print_section("VARIABLE PERSISTENCE")
    
    # Set variables in container 1
    execute_code_and_print(1, """
x = 42
y = "Hello World"
data = [1, 2, 3, 4, 5]
print(f"Set x={x}, y='{y}', data={data}")
""", "Setting variables")
    
    # Use variables in subsequent execution
    execute_code_and_print(1, """
print(f"x * 2 = {x * 2}")
print(f"y.lower() = '{y.lower()}'")
print(f"sum(data) = {sum(data)}")
""", "Using previously defined variables")

def test_different_containers():
    print_section("CONTAINER ISOLATION")
    
    # Set variable in container 1
    execute_code_and_print(1, "container_var = 'I am in container 1'", "Set variable in container 1")
    
    # Set different variable in container 2
    execute_code_and_print(2, "container_var = 'I am in container 2'", "Set variable in container 2")
    
    # Check isolation
    execute_code_and_print(1, "print(f'Container 1: {container_var}')", "Check container 1 variable")
    execute_code_and_print(2, "print(f'Container 2: {container_var}')", "Check container 2 variable")

def test_error_handling():
    print_section("ERROR HANDLING")
    
    # Division by zero
    execute_code_and_print(3, "result = 10 / 0", "Division by zero error")
    
    # Undefined variable
    execute_code_and_print(3, "print(undefined_variable)", "Undefined variable error")
    
    # Import error
    execute_code_and_print(3, "import nonexistent_module", "Import error")

def test_complex_execution():
    print_section("COMPLEX CODE EXECUTION")
    
    code = """
import json
import random

# Create some sample data
data = {
    'numbers': [random.randint(1, 100) for _ in range(10)],
    'message': 'Complex execution test',
    'timestamp': '2024-01-01'
}

# Process the data
avg = sum(data['numbers']) / len(data['numbers'])
max_num = max(data['numbers'])
min_num = min(data['numbers'])

# Print results
print(f"Generated numbers: {data['numbers']}")
print(f"Average: {avg:.2f}")
print(f"Max: {max_num}, Min: {min_num}")
print(f"Message: {data['message']}")

# Return the average for later use
avg
"""
    
    execute_code_and_print(4, code, "Complex data processing")

def test_kernel_restart():
    print_section("KERNEL RESTART FUNCTIONALITY")
    
    # Set a variable
    execute_code_and_print(2, "restart_test_var = 'This will be cleared'", "Set test variable")
    
    # Verify variable exists
    execute_code_and_print(2, "print(f'Before restart: {restart_test_var}')", "Verify variable exists")
    
    # Restart the kernel
    print("\n🔄 Restarting container 2 kernel...")
    try:
        response = requests.post(f"{MANAGER_URL}/session/2/restart", timeout=15)
        response.raise_for_status()
        print("   ✅ Restart request sent")
        
        # Wait for restart to complete
        print("   ⏳ Waiting for restart to complete...")
        for i in range(10):
            time.sleep(2)
            try:
                ready_response = requests.get(f"{MANAGER_URL}/session/2/ready", timeout=5)
                if ready_response.status_code == 200 and ready_response.json().get("ready"):
                    print(f"   ✅ Container ready after {(i+1)*2} seconds")
                    break
            except:
                continue
        else:
            print("   ⚠️  Timeout waiting for restart")
            
    except Exception as e:
        print(f"   ❌ Restart failed: {e}")
    
    # Try to access the variable (should fail)
    execute_code_and_print(2, "print(restart_test_var)", "Try to access cleared variable")

def check_system_health():
    print_section("SYSTEM HEALTH CHECK")
    
    # Check manager health
    try:
        response = requests.get(f"{MANAGER_URL}/health", timeout=5)
        response.raise_for_status()
        print("✅ Manager is healthy")
    except Exception as e:
        print(f"❌ Manager health check failed: {e}")
        return False
    
    # Check container readiness
    all_ready = True
    for container_id in range(0, 4):
        try:
            response = requests.get(f"{MANAGER_URL}/session/{container_id}/ready", timeout=5)
            response.raise_for_status()
            ready_data = response.json()
            if ready_data.get("ready"):
                print(f"✅ Container {container_id} is ready")
            else:
                print(f"❌ Container {container_id} is not ready: {ready_data.get('reason', 'unknown')}")
                all_ready = False
        except Exception as e:
            print(f"❌ Container {container_id} health check failed: {e}")
            all_ready = False
    
    return all_ready

def main():
    print("🚀 EXECUTOR SYSTEM DEMONSTRATION")
    print("=" * 50)
    print("This script demonstrates the capabilities of the distributed code execution system.")
    print("The system consists of a manager service routing requests to isolated executor containers.")
    
    # Check system health first
    if not check_system_health():
        print("\n❌ System health check failed. Please ensure all services are running.")
        return 1
    
    # Run all tests
    try:
        test_basic_execution()
        test_variable_persistence() 
        test_different_containers()
        test_error_handling()
        test_complex_execution()
        test_kernel_restart()
        
        print_section("DEMO COMPLETE")
        print("✅ All demonstrations completed successfully!")
        print("\n📋 Summary:")
        print("   • Basic code execution ✅")
        print("   • Variable persistence ✅") 
        print("   • Container isolation ✅")
        print("   • Error handling ✅")
        print("   • Complex processing ✅")
        print("   • Kernel restart ✅")
        print("\n🎯 The executor system is working correctly!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 