import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros de la simulación
population_size = 1000  # Número de personas en la población

# Parámetros para hombres (media y desviación estándar)
mean_height_men = 175  # Media de la estatura para hombres en cm
std_dev_height_men = 7  # Desviación estándar de la estatura para hombres
mean_weight_men = 78  # Media del peso para hombres en kg
std_dev_weight_men = 10  # Desviación estándar del peso para hombres

# Parámetros para mujeres (media y desviación estándar)
mean_height_women = 162  # Media de la estatura para mujeres en cm
std_dev_height_women = 6  # Desviación estándar de la estatura para mujeres
mean_weight_women = 65  # Media del peso para mujeres en kg
std_dev_weight_women = 8  # Desviación estándar del peso para mujeres

# Proporción de hombres y mujeres
proportion_men = 0.5  # 50% hombres, 50% mujeres

# Generar datos de estaturas y pesos
gender = np.random.choice(['Hombre', 'Mujer'], size=population_size, p=[proportion_men, 1 - proportion_men])
height = np.zeros(population_size)
weight = np.zeros(population_size)

for i in range(population_size):
    if gender[i] == 'Hombre':
        height[i] = np.random.normal(mean_height_men, std_dev_height_men)
        weight[i] = np.random.normal(mean_weight_men, std_dev_weight_men)
    else:
        height[i] = np.random.normal(mean_height_women, std_dev_height_women)
        weight[i] = np.random.normal(mean_weight_women, std_dev_weight_women)

# Crear un DataFrame con los datos generados
population_data = pd.DataFrame({
    'Género': gender,
    'Estatura (cm)': height,
    'Peso (kg)': weight
})

# Mostrar un resumen estadístico de los datos
print(population_data.describe())

# Graficar la distribución de estaturas y pesos
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.hist(height, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de Estaturas')
plt.xlabel('Estatura (cm)')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
plt.hist(weight, bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribución de Pesos')
plt.xlabel('Peso (kg)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()
