"""
CALCULUS FOR MACHINE LEARNING
-----------------------------
This script covers:
1. Derivatives and gradients using SymPy
2. Chain rule demonstration
3. Gradient Descent optimization
4. Multi-variable gradient visualization
5. Neural network-like gradient example
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# =====================================
# 1 Derivatives with SymPy
# =====================================
print("=== 1. Derivatives ===")

x = sp.Symbol('x')
f = x**2 + 3*x + 2
f_prime = sp.diff(f, x)

print("Function:", f)
print("Derivative:", f_prime)
print("f'(2) =", f_prime.subs(x, 2))

# =====================================
# 2️ Partial Derivatives & Gradient
# =====================================
print("\n=== 2. Gradient of Multivariable Function ===")

x, y = sp.symbols('x y')
f_xy = x**2 + y**2 + 3*x*y
df_dx = sp.diff(f_xy, x)
df_dy = sp.diff(f_xy, y)

print("f(x, y) =", f_xy)
print("∂f/∂x =", df_dx)
print("∂f/∂y =", df_dy)

# Gradient vector
grad = [df_dx, df_dy]
print("Gradient =", grad)

# =====================================
# 3️ Chain Rule Example
# =====================================
print("\n=== 3. Chain Rule Example ===")

x = sp.Symbol('x')
g = sp.sin(x)
f = sp.exp(g)
chain_rule_result = sp.diff(f, x)
print("f(g(x)) = e^(sin(x))")
print("Using Chain Rule: df/dx =", chain_rule_result)

# =====================================
# 4️ Gradient Descent Example
# =====================================
print("\n=== 4. Gradient Descent Optimization ===")

def f(x):
    return x**2 + 4*x + 1

def df(x):
    return 2*x + 4

x_old = 10       # initial guess
alpha = 0.1      # learning rate
iterations = 25
x_points = []
f_points = []

for i in range(iterations):
    grad = df(x_old)
    x_new = x_old - alpha * grad
    x_points.append(x_new)
    f_points.append(f(x_new))
    print(f"Iter {i+1}: x={x_new:.4f}, f(x)={f(x_new):.4f}")
    x_old = x_new

# Plot
plt.figure(figsize=(6,4))
plt.plot(x_points, f_points, 'ro-', label='Descent Path')
plt.title("Gradient Descent Optimization")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# =====================================
# 5️  Gradient in Multiple Dimensions
# =====================================
print("\n=== 5. Gradient in 2D Function Visualization ===")

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = X**2 + Y**2 + 3*X*Y

# Compute gradients
dZ_dx = 2*X + 3*Y
dZ_dy = 2*Y + 3*X

plt.figure(figsize=(6,5))
plt.contour(X, Y, Z, 30, cmap='viridis')
plt.quiver(X, Y, -dZ_dx, -dZ_dy, color='r')  # negative gradient direction
plt.title("Gradient Field of f(x,y) = x² + y² + 3xy")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# =====================================
# 6️ Neural Network-like Gradient Example
# =====================================
print("\n=== 6. Backpropagation-style Gradient Example ===")

# Suppose y = sigmoid(w*x + b)
x_val, w_val, b_val = sp.symbols('x_val w_val b_val')

z = w_val * x_val + b_val
y = 1 / (1 + sp.exp(-z))

# Derivative of y w.r.t. w and b (like NN weights)
dy_dw = sp.diff(y, w_val)
dy_db = sp.diff(y, b_val)

print("dy/dw =", dy_dw)
print("dy/db =", dy_db)

# Substitute numerical values
subs_dict = {x_val: 2, w_val: 0.5, b_val: 0.1}
print("dy/dw (values) =", dy_dw.subs(subs_dict))
print("dy/db (values) =", dy_db.subs(subs_dict))

print("\n All Calculus concepts demonstrated successfully!")
