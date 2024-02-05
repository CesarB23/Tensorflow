from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import uuid
#flask --app "nombre" run --debug
# o if name == main python app
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = keras.models.load_model("modelo_completo.h5")

classes = {
        0: "Glioma",
        1: "Meningioma",
        2: "No Tumor",
        3: "Pituitario"
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Carga la imagen enviada desde la solicitud
    image = request.files['file']
    
    # Genera un nombre único para la imagen
    image_filename = str(uuid.uuid4()) + '.jpg'
    
    # Guarda la imagen en la carpeta de subidas
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    
    # Ruta de la imagen completa
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    
    # Lee la imagen y realiza el preprocesamiento necesario
    img = keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict(img_array)

    # Obtiene la clase predicha
    predicted_class_name = classes[np.argmax(prediction)]

    # Renderiza la plantilla de resultados con la imagen y su clasificación
    return render_template("index.html", image_path=image_path, predicted_class=predicted_class_name)

if __name__ == "__main__":
    app.run(debug=True)

