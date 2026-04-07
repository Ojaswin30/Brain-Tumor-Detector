import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_module = importlib.import_module("tensorflow.keras.preprocessing.image")
ImageDataGenerator = image_module.ImageDataGenerator
# from tensorflow.keras.models import Sequential
Sequential_module = importlib.import_module("tensorflow.keras.models")
Sequential = Sequential_module.Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
Layers_module = importlib.import_module("tensorflow.keras.layers")
Conv2D = Layers_module.Conv2D
MaxPooling2D = Layers_module.MaxPooling2D
Dense = Layers_module.Dense
Flatten = Layers_module.Flatten
Dropout = Layers_module.Dropout
BatchNormalization = Layers_module.BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
callbacks_module = importlib.import_module("tensorflow.keras.callbacks")
EarlyStopping = callbacks_module.EarlyStopping
ReduceLROnPlateau = callbacks_module.ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "Datasets"   # make sure this contains /yes and /no folders

# =========================
# CHECK DATASET
# =========================
print("Checking dataset structure...")
print("Classes found:", os.listdir(DATA_DIR))

# =========================
# DATA GENERATOR (AUGMENTATION + NORMALIZATION)
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class indices:", train_data.class_indices)

# =========================
# MODEL BUILDING
# =========================
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS (ANTI-OVERFITTING)
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    verbose=1
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce]
)

# =========================
# SAVE MODEL
# =========================
model.save("brain_tumor_model.h5")
print("Model saved as brain_tumor_model.h5")

# =========================
# PLOT ACCURACY & LOSS
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

# =========================
# EVALUATION (IMPORTANT)
# =========================
val_data.reset()

predictions = model.predict(val_data)
predictions = (predictions > 0.5).astype(int)

true_labels = val_data.classes

print("\nClassification Report:")
print(classification_report(true_labels, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions))