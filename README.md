# SciPy Guide: Scientific Computing in Python

## Overview
SciPy (Scientific Python) is a collection of mathematical algorithms and convenience functions built on NumPy. It provides advanced capabilities for optimization, linear algebra, integration, interpolation, and other domains.

## Core Features and Submodules

### 1. scipy.optimize - Optimization and Root Finding
Used for finding minima/maxima of functions and solving equations.

```python
from scipy import optimize
import numpy as np

# Example: Finding minimum of a function
def f(x):
    return x**2 + 10*np.sin(x)

result = optimize.minimize(f, x0=0)  # Start search at x=0
print(f"Minimum found at x = {result.x}")
```

### 2. scipy.integrate - Integration
Provides tools for numerical integration.

```python
from scipy import integrate

# Example: Definite integral of sin(x) from 0 to pi
def integrand(x):
    return np.sin(x)

result, error = integrate.quad(integrand, 0, np.pi)
print(f"Integral = {result:.6f}, Error = {error:.6f}")
```

### 3. scipy.interpolate - Interpolation
Creates functions based on discrete data points.

```python
from scipy import interpolate

# Example: Creating smooth curve through points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 7, 4])

# Create interpolation function
f = interpolate.interp1d(x, y, kind='cubic')

# Generate smooth curve
x_new = np.linspace(0, 5, 100)
y_new = f(x_new)
```

### 4. scipy.linalg - Linear Algebra
Advanced linear algebra operations.

```python
from scipy import linalg

# Example: Solving system of linear equations
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

x = linalg.solve(A, b)
print(f"Solution: {x}")

# Computing eigenvalues
eigenvals = linalg.eigvals(A)
print(f"Eigenvalues: {eigenvals}")
```

### 5. scipy.stats - Statistical Functions
Comprehensive collection of probability distributions and statistical functions.

```python
from scipy import stats

# Example: Generating normal distribution and calculating statistics
data = stats.norm.rvs(loc=0, scale=1, size=1000)
ks_statistic, p_value = stats.kstest(data, 'norm')
print(f"KS test p-value: {p_value}")

# Performing t-test
sample1 = stats.norm.rvs(loc=0, scale=1, size=100)
sample2 = stats.norm.rvs(loc=0.5, scale=1, size=100)
t_stat, p_val = stats.ttest_ind(sample1, sample2)
```

### 6. scipy.signal - Signal Processing
Tools for signal processing.

```python
from scipy import signal

# Example: Creating and applying a filter
t = np.linspace(0, 1, 1000)
raw_signal = np.sin(2*np.pi*10*t) + np.random.normal(0, 0.1, len(t))

# Design filter
b, a = signal.butter(4, 0.2)  # 4th order Butterworth filter
filtered_signal = signal.filtfilt(b, a, raw_signal)
```

## Practical Examples

### Example 1: Curve Fitting
```python
from scipy import optimize

# Generate sample data with noise
x_data = np.linspace(0, 10, 20)
y_data = 3 * np.exp(-x_data/2) + np.random.normal(0, 0.1, len(x_data))

# Define model function
def model(x, a, b):
    return a * np.exp(-b * x)

# Fit model to data
popt, pcov = optimize.curve_fit(model, x_data, y_data)
print(f"Fitted parameters: a={popt[0]:.2f}, b={popt[1]:.2f}")
```

### Example 2: Finding Roots
```python
from scipy import optimize

# Define equation: x^3 - 2x^2 + 4x - 8 = 0
def equation(x):
    return x**3 - 2*x**2 + 4*x - 8

# Find root using different methods
root1 = optimize.newton(equation, x0=0)  # Newton's method
root2 = optimize.bisect(equation, 0, 5)  # Bisection method
print(f"Root (Newton): {root1:.4f}")
print(f"Root (Bisect): {root2:.4f}")
```

### Example 3: Statistical Analysis
```python
from scipy import stats

# Generate two samples
group1 = stats.norm.rvs(loc=10, scale=2, size=100)
group2 = stats.norm.rvs(loc=12, scale=2, size=100)

# Perform statistical tests
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-test p-value: {p_value}")

# Calculate confidence interval
confidence_interval = stats.t.interval(0.95, len(group1)-1,
                                     loc=np.mean(group1),
                                     scale=stats.sem(group1))
```

## Best Practices

1. Always import NumPy alongside SciPy:
```python
import numpy as np
from scipy import [submodule]
```

2. Use specific imports for better code readability:
```python
from scipy.optimize import minimize  # Better than scipy.optimize.minimize
```

3. Handle errors and convergence:
```python
try:
    result = optimize.minimize(f, x0=0)
    if result.success:
        print(f"Optimization successful: {result.x}")
    else:
        print(f"Optimization failed: {result.message}")
except Exception as e:
    print(f"Error occurred: {e}")
```

4. Set random seeds for reproducibility:
```python
np.random.seed(42)
data = stats.norm.rvs(size=1000)
```
