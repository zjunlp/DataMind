import requests

INSTANCE_URL = "http://localhost:60000"

def test_code_folder_loading():
    """Test that code from the test_code folder is properly loaded and available in the kernel"""
    
    # Test that functions from calculations.py are available
    resp1 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "result = calculate_circle_area(5)"})
    assert resp1.status_code == 200
    
    # Verify the calculation result
    resp2 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "print(f'Circle area: {result}')"})
    assert resp2.status_code == 200
    outputs = resp2.json()["outputs"]
    assert any("Circle area: 78.53981633974483" in str(output) for output in outputs)
    
    # Test function from utils.py
    resp3 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "greeting = greet('World')"})
    assert resp3.status_code == 200
    
    # Verify the greeting result
    resp4 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "print(greeting)"})
    assert resp4.status_code == 200
    outputs = resp4.json()["outputs"]
    assert any("Hello, World!" in str(output) for output in outputs)
    
    # Test factorial function
    resp5 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "fact_result = factorial(5)"})
    assert resp5.status_code == 200
    
    # Verify factorial result
    resp6 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "print(f'Factorial of 5: {fact_result}')"})
    assert resp6.status_code == 200
    outputs = resp6.json()["outputs"]
    assert any("Factorial of 5: 120" in str(output) for output in outputs)
    
    # Test add_numbers function
    resp7 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "sum_result = add_numbers(10, 15)"})
    assert resp7.status_code == 200
    
    # Verify addition result
    resp8 = requests.post(f"{INSTANCE_URL}/execute", 
                          json={"code": "print(f'Sum: {sum_result}')"})
    assert resp8.status_code == 200
    outputs = resp8.json()["outputs"]
    assert any("Sum: 25" in str(output) for output in outputs)
    
    print("✓ Code folder loading test passed")

if __name__ == "__main__":
    test_code_folder_loading()
    print("Code folder test completed!") 