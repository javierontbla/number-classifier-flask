# importar librerías
from flask import Flask, request 
from flask_cors import CORS
import joblib
import numpy as np
from scipy.ndimage import rotate

# cargar modelo ya entrenado
rnd_clf_model = joblib.load('rnd_clf.pkl')

app = Flask(__name__)
CORS(app)

# la función route() es un decorador de la clase flask
# y nos dice a que URL llamar y que acción (POST)
@app.route('/', methods=['POST'])
def predict_react_number():
    if request.method == 'POST': # atrapar POST requests
        react_number = request.get_json() # pasar a JSON
        react_number_np = np.array(react_number) # convertir arr de python en una arr de numpy
        react_number_reshaped = react_number_np.reshape(28, 28) # redimensionar a 28 x 28
        react_number_flip = np.flip(react_number_reshaped, 0) # voltear el arr
        react_number_rotated = rotate(react_number_flip, angle=-90) # rotar -90 grados el arr
        react_number_clean = np.reshape(react_number_rotated, (1,784)) # volver a redimensionar a 1 x 784
        react_number_prediction = rnd_clf_model.predict(react_number_clean) # hacer predicción
        return f"{react_number_prediction[0]}" # enviar resultado de regreso al usuario
    return 'python-server-v1'
  
# iniciar servidor
if __name__ == '__main__':
    app.run()