def fibonacci(n):
    """Calculates the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

if __name__ == "__main__":
    num = int(input("Enter a non-negative integer to find its Fibonacci number: "))
    result = fibonacci(num)
    print(f"The Fibonacci number at position {num} is {result}")