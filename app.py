import numpy as np
import math

def f(x): return x**2 + 5
def df(x): return 2*x
def ddf(x): return 2

def g(x): return math.exp(0.5*x)
def dg(x): return 0.5*math.exp(0.5*x)
def ddg(x): return 0.25*math.exp(0.5*x)

def find_distance_newton(x0, y0, func, dfunc, ddfunc, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    x = initial_guess
    path_x = [x]
    
    for i in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        ddfx = ddfunc(x)
        
        d_p = 2*(x - x0) + 2*(fx - y0)*dfx
        d_pp = 2 + 2*(dfx**2) + 2*(fx - y0)*ddfx
        
        step = d_p / d_pp
        x_next = x - step
        path_x.append(x_next)
        
        if abs(x_next - x) < tolerance:
            x = x_next
            break
        x = x_next
        
    dist = math.sqrt((x - x0)**2 + (func(x) - y0)**2)
    return dist, x, path_x

def secant_search(x0, y0, func, dfunc, x1, x2, tol=1e-7, max_iter=100):
    def dp(val):
        return 2*(val - x0) + 2*(func(val) - y0)*dfunc(val)
        
    history = [(x1, x2)]
    for _ in range(max_iter):
        f1 = dp(x1)
        f2 = dp(x2)
        
        if abs(f2 - f1) < 1e-14:
            break
            
        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        history.append((x2, x_new))
        
        if abs(x_new - x2) < tol:
            x2 = x_new
            break
            
        x1 = x2
        x2 = x_new
        
    dist = math.sqrt((x2 - x0)**2 + (func(x2) - y0)**2)
    return dist, x2, history

data_x = np.array([0, 2, 1, 3])
data_y = np.array([0.5, 3.5, 1.5, 7.5])

def fit_line_coordinate_descent(x, y, iters=10):
    m, c = 0.0, 0.0
    for _ in range(iters):
        c = np.mean(y - m*x)
        m = np.sum(x*(y - c)) / np.sum(x**2)
    mse = np.mean((m*x + c - y)**2)
    return m, c, mse

def fit_parabola_coordinate_descent(x, y, iters=10):
    a, c = 0.0, 0.0
    x_sq = x**2
    for _ in range(iters):
        c = np.mean(y - a*x_sq)
        a = np.sum(x_sq*(y - c)) / np.sum(x**4)
    mse = np.mean((a*x_sq + c - y)**2)
    return a, c, mse

def analytical_pseudo_inverse(A, b):
    return np.linalg.pinv(A) @ b

if __name__ == "__main__":
    
    print("--- Part 1A: Shortest Distance (Polynomial) ---")
    points = [(0, 0), (2, 0), (-4, 0), (6, 0), (-8, 0)]
    for pt in points:
        d_n, x_n, _ = find_distance_newton(pt[0], pt[1], f, df, ddf, initial_guess=pt[0])
        d_s, x_s, _ = secant_search(pt[0], pt[1], f, df, pt[0]-1, pt[0]+1)
        print(f"Target {pt}: Newton x* = {x_n:7.4f} | Secant x* = {x_s:7.4f} | Dist = {d_n:.4f}")
        
    print("\n--- Part 1B: Shortest Distance (Non-Polynomial) ---")
    for pt in [(2, 0), (0, 0)]:
        d_n, x_n, _ = find_distance_newton(pt[0], pt[1], g, dg, ddg, initial_guess=pt[0])
        print(f"Target {pt}: Optimal x* = {x_n:7.4f} | Dist = {d_n:.4f}")

    print("\n--- Part 2: MSE Curve Fitting ---")
    m, c, mse_line = fit_line_coordinate_descent(data_x, data_y)
    a, c_p, mse_parab = fit_parabola_coordinate_descent(data_x, data_y)
    
    A_line = np.column_stack((data_x, np.ones_like(data_x)))
    theta_line = analytical_pseudo_inverse(A_line, data_y)
    
    A_parab = np.column_stack((data_x**2, np.ones_like(data_x)))
    theta_parab = analytical_pseudo_inverse(A_parab, data_y)
    
    print(f"Line Fit (Numerical):  y = {m:.2f}x + {c:.2f}      (MSE: {mse_line:.3f})")
    print(f"Line Fit (Analytical): y = {theta_line[0]:.2f}x + {theta_line[1]:.2f}")
    print(f"\nParab Fit (Numerical): y = {a:.2f}x^2 + {c_p:.2f}    (MSE: {mse_parab:.3f})")
    print(f"Parab Fit (Analytical):y = {theta_parab[0]:.2f}x^2 + {theta_parab[1]:.2f}")