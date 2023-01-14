import os
import tensorflow as tf
import numpy as np
import skimage
from keras_preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "model_uas.h5")

# Preprocess an image
def classify(model,image):
    img = load_img(image, target_size = (150,150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size = 16)
    
    label = 'Daun Nangka' if classes[0][0] > 0 else 'Daun Kemangi'
    classified_prob = classes[0][0] if classes[0][0] > 0 else 1 - classes[0][0]       
    return label, classified_prob


# home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/about", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            file = request.files["image"]
            upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(upload_image_path)
            file.save(upload_image_path)
        except FileNotFoundError:
            return render_template("index.html")    
        label,prob = classify(cnn_model, upload_image_path)
        prob = round((prob*100),2)
        
    return render_template(
        "about.html", image_file_name=file.filename, label=label, prob=prob
    )

@app.route("/about/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True
