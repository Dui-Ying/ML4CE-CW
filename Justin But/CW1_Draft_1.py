import numpy as np
import math
import random

# RBF Kernel
# Length Scale
# Sigma F maybe Sigma Variance
# Noise Variance
def rbf_kernel(x1, x2, length_scale=0.1, sigma_f=1.0):
    return sigma_f**2 * math.exp(-((x1 - x2)**2) / (2 * length_scale**2))

# Build covariance matrix K(X,X)
def cov_matrix(X, noise=1e-10, sigma_f=1.0, length_scale=0.1):
    n = len(X)
    K = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = rbf_kernel(X[i], X[j], length_scale, sigma_f)
            if i == j:
                K[i][j] += noise
    return K

# Compute the inverse of a matrix (using numpy)
def mat_inv(M):
    # Convert to numpy array
    A = np.array(M, dtype=float)
    
    # Use numpy.linalg.inv for matrix inversion
    A_inv = np.linalg.inv(A)
    
    # Convert back to a regular Python list if needed
    return A_inv.tolist()

# GP prediction for mean and variance at a new point x_star
def gp_predict(x_star, X, y, noise=1e-10, sigma_f=1.0, length_scale=0.1):
    K = cov_matrix(X, noise=noise, length_scale=length_scale, sigma_f=sigma_f)
    K_inv = mat_inv(K)
    k_star = [rbf_kernel(xi, x_star, length_scale, sigma_f) for xi in X]
    # Mean
    mu_star = 0.0
    for i in range(len(X)):
        for j in range(len(X)):
            mu_star += k_star[i] * K_inv[i][j] * y[j]
    # Variance
    k_star_star = rbf_kernel(x_star, x_star, length_scale, sigma_f)
    var_star = k_star_star
    for i in range(len(X)):
        for j in range(len(X)):
            var_star -= k_star[i] * K_inv[i][j] * k_star[j]
    # Ensure non-negative due to numerical issues
    var_star = max(var_star, 1e-15)
    return mu_star, var_star

# PDF of standard normal
def phi(z):
    return (1.0/math.sqrt(2*math.pi))*math.exp(-0.5*z*z)

# CDF of standard normal (using error function approximation)
# Consider using a better CDF
def Phi(z):
    return 0.5*(1 + math.erf(z/math.sqrt(2)))

# Expected Improvement
def expected_improvement(x, X, y, f_best, length_scale=0.1, sigma_f=1.0):
    mu, var = gp_predict(x, X, y, length_scale, sigma_f)
    sigma = math.sqrt(var)
    if sigma < 1e-15:
        # No uncertainty, improvement if mu > f_best
        return max(0.0, f_best - mu)  # If we are minimizing, improvement is best - mu
    # For minimization: improvement = f_best - mu
    imp = f_best - mu
    Z = imp / sigma
    ei = imp * Phi(Z) + sigma * phi(Z)
    return ei

# Bayesian Optimization loop
# Iteration + Initial Points
def bayesian_optimization(objective, bounds=(0.0, 1.0), n_iter=10, init_points=3):
    # Initial random points
    X = [random.uniform(bounds[0], bounds[1]) for _ in range(init_points)]
    y = [objective(xi) for xi in X]
    
    for iteration in range(n_iter):
        f_best = min(y)
        # Search a set of candidates (naive approach) - in practice, use a smarter optimizer
        candidates = [bounds[0] + (bounds[1]-bounds[0])*i/100.0 for i in range(101)]
        
        best_ei = -1
        best_x = None
        for c in candidates:
            ei = expected_improvement(c, X, y, f_best)
            if ei > best_ei:
                best_ei = ei
                best_x = c
        
        # Evaluate objective at the best candidate
        new_y = objective(best_x)
        X.append(best_x)
        y.append(new_y)

        print(f"Iteration {iteration+1}: Best x = {best_x:.4f}, f(x) = {new_y:.4f}, Current best f = {min(y):.4f}")
    
    return X, y