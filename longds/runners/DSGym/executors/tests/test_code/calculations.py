import math

def calculate_circle_area(radius):
    '''Calculate the area of a circle'''
    return math.pi * radius ** 2

def factorial(n):
    '''Calculate factorial of n'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("calculations.py loaded successfully!") 