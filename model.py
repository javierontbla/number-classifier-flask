# importar librerías de Machine Learning
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import joblib

print("entrenando...")
# descargar conjunto de datos (dataset)
# que contiene 70,000 números escritos a mano
mnist = fetch_openml('mnist_784', version=1)

# dividir en atributos y etiquetas
X = mnist["data"]
y = mnist["target"].astype(np.uint8) # convertir str a int

# si el pixel es mayor a 1, reemplazar con 255 
# para tener un pixel completamente negro
X_all_black = X.replace([range(1, 255)], 255)

# declarar instancia de RandomForest y entrenar el modelo
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_all_black.values, y)

# guardar modelo en un archivo .pkl
joblib.dump(rnd_clf, 'rnd_clf.pkl')
print("entrenamiento completado")