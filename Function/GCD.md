# GCD
- Greatest Common Divisor.
- Implement using Euclidean algorithm.

# Implementation
```pyhton
def GCD(a, b):
    if b == 0:
        return a
    else:
        return GCD(b, a%b)
```
