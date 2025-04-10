# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:02:11 2025

@author: Amogh Kulkarni
"""

# Import necessary packages
import numpy as np 
import matplotlib.pyplot as plt

# Parameters for true signal x*
vector_length = 4000      # Length of the signal
discont = 10              # Number of discontinuities
np.random.seed(2)         # Seed for reproducibility
upperbound = 5            # Upper bound for signal values
lowerbound = -5           # Lower bound for signal values

# Generate sorted discontinuity positions with start (0) and end (vector_length)
discont_positions = np.sort(
    np.append(
        np.random.choice(np.arange(1, vector_length - 1), size=discont, replace=False),
        [0, vector_length]
    )
)

# Generate random constant values for each segment between discontinuities
values = lowerbound + (upperbound - lowerbound) * np.random.rand(discont + 1)

# Generate true signal x* as piecewise constant
x_star = np.zeros(vector_length)
for i in range(discont + 1):
    x_star[discont_positions[i]:discont_positions[i+1]] = values[i]

# Generate Noisy Data b by adding Gaussian noise
b = x_star + np.random.normal(0, 1, vector_length)

# Parameters for Minimising Total Variation
Lambda = 10          # Regularization parameter
Iterations = 5000    # Number of gradient descent iterations
x_estimate = b       # Initialize estimate with the noisy signal

def D(x):
    """
    Forward difference operator.
    
    Parameters:
        x (ndarray): Input signal of length N.
    
    Returns:
        ndarray: Differences of length N-1.
    """
    return np.diff(x)

def DT(x):
    """
    Transpose (adjoint) of the forward difference operator.
    Implements a discrete divergence operation.
    
    Parameters:
        x (ndarray): Input gradient of length N-1.
    
    Returns:
        ndarray: Divergence result of length N.
    """
    return np.concatenate(([-x[0]], -np.diff(x), [x[-1]]))    

# Gradient Descent loop for Total Variation Denoising
for n in range(1, Iterations):
    x_estimate = x_estimate - (1/n) * ((x_estimate - b) + Lambda * DT(np.sign(D(x_estimate))))

# Plot 1: True signal x*
plt.figure(figsize=(12, 3))
plt.plot(x_star, label='True Signal $x^*$', linewidth=2)
plt.title("True Signal $x^*$")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Plot 2: Noisy signal b
plt.figure(figsize=(12, 3))
plt.plot(b, label='Noisy Observation $b$', alpha=0.6)
plt.title("Noisy Observation $b$")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Plot 3: Estimated signal x̂
plt.figure(figsize=(12, 3))
plt.plot(x_estimate, label='Estimated Signal $\hat{x}$', linewidth=2)
plt.title("Estimated Signal $\hat{x}$ after TV Denoising")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
