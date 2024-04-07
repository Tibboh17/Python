# isPrime
- A function that checks if the given number is prime number.
- Implment by using division or the sieve of Eratosthenes.

# Implemetation with Division
```python
from math import sqrt

def isPrime(num):
    if num == 0 or num == 1:
        return False
    if num == 2:
        return True
    for i in range(2, int(sqrt(num))+2):
        if num%i == 0:
            return False
    return True
```

# Implementation with the Sieve of Eratosthenes
```python
from math import sqrt

def Eratos(num):
    n = 1000
    array = [True for _ in range(n+1)]
    array[0], array[1] = False, False
    
    for i in range(2, int(sqrt(n))+2):
        if array[i]:
            j = 2
            while i*j <= n:
                array[i*j] = False
                j += 1
    return array[num]
```
