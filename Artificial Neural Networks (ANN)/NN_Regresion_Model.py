import pandas as pd
import numpy as np
import os.path
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

list = []
resultados = pd.read_csv("Resultados.csv")
data1 = pd.read_csv("Dataset/C011180098.txt", sep="\t", header=None)
data2 = pd.read_csv("Dataset/C011180099.txt", sep="\t", header=None)
list.append(data1[1])
list.append(data2[1])
for x in range(100, 454):
    posicion = str(x)
    ruma = "C011180"+posicion
    ruta = "Dataset/"+ruma+".txt"
    if os.path.isfile(ruta) == True:
        data = pd.read_csv(ruta, sep="\t", header=None)
        list.append(data[1])
        
array_fishmeal = np.asarray(list)
resultados_proteina = resultados["PROTEINA (DUMAS)"]
array_proteina = np.asarray(resultados_proteina)

### Entrenamiento con Bandas hiperespectrales optimas
# bandas_optimas0 = np.array([65,67,68,69,70,72,79,80,81,82,83,84,85,86,87,88,132,133,143,170,172,176,177,178,229,234])
# num_ban_op= len(bandas_optimas0)
# bandas_optimas= bandas_optimas0-1
# array_fishmeal=array_fishmeal[:,bandas_optimas]

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(array_fishmeal, array_proteina, test_size = 0.2, random_state=0)

# Preprocesamiento de datos
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# RNA con keras
model = Sequential()
model.add(Dense(units =10, kernel_initializer="uniform", activation="relu", input_dim = num_ban_op))
#model.add(Dropout(0.2)) #agregar dropout a cada capa
model.add(Dense(units =25, kernel_initializer="uniform", activation="relu"))
#model.add(Dropout(0.1))
model.add(Dense(units =10, kernel_initializer="uniform", activation="relu"))
model.add(Dense(units =1, kernel_initializer="uniform", activation="relu"))
model.compile(optimizer = "adam", loss="mean_squared_error", metrics=["mape"])
#model.compile(optimizer = "sdg", loss=tf.keras.losses.mean_squared_error() )

# EarlyStopping
callbacks = [EarlyStopping(monitor="loss", patience=10)]


#modelo = model.fit(X_train, y_train, batch_size = 50, epochs = 1000, 
                  #validation_data=(X_test, y_test))
modelo = model.fit(X_train, y_train, batch_size = 30, epochs = 1000, 
          validation_data=(X_test, y_test), callbacks=callbacks)

y_pred = model.predict(X_test)

#plt.plot(modelo.history["loss"], label="perdida")
#plt.plot(modelo.history["mape"], label="error absoluto porcentual")

loss = modelo.history["loss"]
val_loss = modelo.history["val_loss"]
epoch = range(1, len(loss)+1)
p1=plt.plot(epoch, loss, "y", label = "Training loss")
plt.plot(epoch, val_loss, "r", label = "Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#comentar los gráficos que no quieres que se muestren

p1=plt.plot(epoch, np.log10(loss), "y", label = "Training loss")
plt.plot(epoch, np.log10(val_loss), "r", label = "Validation loss")
plt.title("Log loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show(p1)

y1_pred = y_pred[:,0]        
mse = mean_squared_error(y_test,y1_pred)
mseL=str(mse)
p3=plt.plot(y_test,y1_pred,label='MSE='+mseL,marker='*',linestyle="None", color='b')
plt.title("")
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.legend()                       # Mostramos la leyenda automáticamente
plt.show(p3) 

plt.scatter(y_train,model.predict(X_train)[:,0]-y_train)
plt.scatter(y_test,model.predict(X_test)[:,0]-y_test)
plt.xlabel("True value")
plt.ylabel("Error")
plt.legend(["Train","Test"])


#from sklearn.externals import joblib
#joblib.dump(model, 'modelo_entrenadoDEFINITVO.pkl')