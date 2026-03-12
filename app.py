from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("waste_model.h5")

classes = ["cardboard","glass","metal","paper","plastic","trash"]

@app.route("/", methods=["GET","POST"])
def index():
    label = ""

    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join("static", file.filename)
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img,(128,128))
        img = img/255.0
        img = np.expand_dims(img,axis=0)

        pred = model.predict(img)
        label = classes[np.argmax(pred)]

        return render_template("index.html",label=label,image=path)

    return render_template("index.html",label=label)

if __name__ == "__main__":
    app.run()