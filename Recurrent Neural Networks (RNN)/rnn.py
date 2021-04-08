# Parte 1 - Preprocesado de los datos

# Importación de las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Ventas_periodos_Train.csv")
training_set  = dataset_train.iloc[:, 1:2].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps (2 meses de ventas) y 1 salida
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Redimensión de los datos
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Parte 2 - Construcción de la RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Inicialización del modelo
regressor = Sequential()

# Añadir la primera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))

# Añadir la segunda capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Añadir la capa de salida
regressor.add(Dense(units = 1))

# Compilar la RNR
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Ajustar la RNR al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Obtener el valor de las ventas reales
dataset_test = pd.read_csv('Ventas_periodos_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Obtener la predicción de las ventas con la RNR
dataset_total = pd.concat((dataset_train['Ventas'], dataset_test['Ventas']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizar los Resultados
plt.plot(real_stock_price, color = 'red', label = 'Cantidad real de ventas en una tienda por periodo')
plt.plot(predicted_stock_price, color = 'blue', label = 'Cantidad predicha de ventas en una tienda por periodo')
plt.title("Predicción de ventas de una tienda para reducir-aumentar stock de productos.")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de ventas")
plt.legend()
plt.show()






