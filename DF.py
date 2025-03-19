import numpy as np
import matplotlib.pyplot as plt

def solve_poisson(f, u0, u1, N):
    h = 1 / (N + 1)
    x = np.linspace(0, 1, N+2)
    
    # Construction de la matrice A
    A = np.diag(2 * np.ones(N)) + np.diag(-1 * np.ones(N-1), 1) + np.diag(-1 * np.ones(N-1), -1)
    A = A / h**2
    
    # Construction du vecteur b
    b = np.array([f(xi) for xi in x[1:-1]])
    b = b * h**2
    b[0] += u0 / h**2
    b[-1] += u1 / h**2
    
    # Résolution du système linéaire
    u = np.linalg.solve(A, b)
    
    # Ajout des conditions aux limites
    u = np.concatenate(([u0], u, [u1]))
    
    return x, u

def f1(x):
    return np.pi**2 * np.sin(np.pi * x)

def f2(x):
    return 6 * x

# Conditions aux limites
u0 = 0
u1 = 0

# Maillages
N_values = [10, 20, 40, 80, 160, 320]

# Solutions exactes
def u_exact1(x):
    return np.sin(np.pi * x)

def u_exact2(x):
    return x**3

# Calcul des erreurs
errors1 = []
errors2 = []

for N in N_values:
    x, u = solve_poisson(f1, u0, u1, N)
    u_exact = u_exact1(x)
    error = np.max(np.abs(u - u_exact))
    errors1.append(error)
    
    x, u = solve_poisson(f2, u0, u1, N)
    u_exact = u_exact2(x)
    error = np.max(np.abs(u - u_exact))
    errors2.append(error)

# Calcul de l'ordre de convergence
h_values = [1 / (N + 1) for N in N_values]

order1 = np.polyfit(np.log(h_values), np.log(errors1), 1)[0]
order2 = np.polyfit(np.log(h_values), np.log(errors2), 1)[0]

print(f"Ordre de convergence pour u(x) = sin(pi*x): {order1}")
print(f"Ordre de convergence pour u(x) = x^3: {order2}")

# Représentation graphique
plt.figure(figsize=(12, 6))

# Solution pour u(x) = sin(pi*x)
x, u = solve_poisson(f1, u0, u1, 20)
u_exact = u_exact1(x)
plt.subplot(2, 2, 1)
plt.plot(x, u, 'b-', label='Solution approchée')
plt.plot(x, u_exact, 'r--', label='Solution exacte')
plt.title('Solution pour u(x) = sin(pi*x)')
plt.legend()

# Erreur pour u(x) = sin(pi*x)
plt.subplot(2, 2, 2)
plt.plot(x, np.abs(u - u_exact), 'g-', label='Erreur')
plt.title('Erreur pour u(x) = sin(pi*x)')
plt.legend()

# Solution pour u(x) = x^3
x, u = solve_poisson(f2, u0, u1, 20)
u_exact = u_exact2(x)
plt.subplot(2, 2, 3)
plt.plot(x, u, 'b-', label='Solution approchée')
plt.plot(x, u_exact, 'r--', label='Solution exacte')
plt.title('Solution pour u(x) = x^3')
plt.legend()

# Erreur pour u(x) = x^3
plt.subplot(2, 2, 4)
plt.plot(x, np.abs(u - u_exact), 'g-', label='Erreur')
plt.title('Erreur pour u(x) = x^3')
plt.legend()

plt.tight_layout()
plt.show()