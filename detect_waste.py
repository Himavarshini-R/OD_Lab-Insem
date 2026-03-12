import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("waste_model.h5")

# Classes
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Read test image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# Preprocess image
img_resized = cv2.resize(img,(128,128))
img_resized = img_resized/255.0
img_resized = np.expand_dims(img_resized, axis=0)

# Predict
prediction = model.predict(img_resized)

label = classes[np.argmax(prediction)]

print("Detected Waste Type:",label)

# Display result
cv2.putText(img,label,(20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,(0,255,0),2)

cv2.imshow("Waste Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()