import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


model=load_model('BrainTumor10epochs.h5')

image_path = input("Enter the image file path: ")

# Read and preprocess the input image
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found or invalid image path.")
else:
    img=Image.fromarray(image)

    img=img.resize((64,64)) 

    img=np.array(img)

    input_img=np.expand_dims(img, axis=0)
    result = model.predict(input_img)
    predicted_class = (result > 0.5).astype(int)


    if predicted_class == 0:
        print("No tumor detected")
    else:
        print("Tumor detected")