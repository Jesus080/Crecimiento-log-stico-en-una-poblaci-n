import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
r = 0.1  # Tasa de crecimiento (10% por unidad de tiempo)
K = 1000  # Capacidad de carga (número máximo de individuos)
P0 = 10   # Población inicial
time_steps = 100  # Número de pasos de tiempo
dt = 1    # Incremento de tiempo

# Inicialización de la población
population = np.zeros(time_steps)
population[0] = P0

# Simulación del crecimiento logístico
for t in range(1, time_steps):
    P_t = population[t-1]
    dP = r * P_t * (1 - P_t / K)  # Variación de la población
    population[t] = P_t + dP * dt

# Graficar el crecimiento de la población
plt.figure(figsize=(10, 6))
plt.plot(np.arange(time_steps), population, label="Población", color='blue')
plt.axhline(K, color='red', linestyle='--', label=f'Capacidad de carga (K={K})')
plt.title('Crecimiento Logístico de una Población')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.grid(True)
plt.show()
