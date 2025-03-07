import cv2                                                                                         #used to read, resize and process images
import os                                                                                          #Used to extract images from the folders
from PIL import Image                                                                              #used to convert different imaage formats which may not be accepted by cv2 to acceptable formats
import numpy as np                                                                                 #neural networks expect numpy arrays as an input to make calculations easier

from sklearn.model_selection import train_test_split                                               #used to split the dataset into train dataset and test dataset

import tensorflow as tf
from tensorflow import keras                                                                       #Main framework for building and training the cnn model

from tensorflow.keras.utils import normalize                                                       #used to normalize pixel values (scales between 0 and 1) for better training efficiancy
from tensorflow.keras.models import Sequential                                                     #defines the cnn model as sequential stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

#conv2D - convulational layer that extracts features from mri images
#MaxPooling2D - downsamples feature maps to reduce computation and prevent overfitting
#Activation - adds non-linearity (eg. ReLU) to layers
#Dropout - Randomly deactivates neurons during training to prevent overfitting
#flatten - Converts 2D feature maps into a 1D vector for the Dense Layers
#Dense - Fully Connected layers that make final classification predictions

from keras.utils import to_categorical                                                             #converts class labels into one-hot encoding. needed for categorical classification task

image_directory='Datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE=64
#print(no_tumor_images)

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)



for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)



#convert dataset to numpy array
dataset=np.array(dataset)
label=np.array(label)



x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)




x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# y_train=to_categorical(y_train, num_classes=2)
# y_test=to_categorical(y_test, num_classes=2)


# Model Building

model=Sequential()

model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2,)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2,)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2,)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test,y_test), shuffle=False)

model.save('BrainTumor10epochs.h5')
