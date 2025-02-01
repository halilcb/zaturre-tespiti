# Import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import os

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# %% Load Data

labels = ["PNEUMONIA", "NORMAL"]
img_size = 150

def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in tqdm(os.listdir(path)):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print("Read image error")
                    continue
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print("Error:", e)
    
    return np.array(data, dtype=object)


train = get_training_data("zaturre-tespiti/chest_xray/chest_xray/train")
test = get_training_data("zaturre-tespiti/chest_xray/chest_xray/test")
val = get_training_data("zaturre-tespiti/chest_xray/chest_xray/val")

# %% data visualization and preprocessing

l = []
for i in train:
    if i[1] == 0:
        l.append("PNEUMONIA")
    else:
        l.append("NORMAL")

sns.countplot(x=l)

x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

plt.figure()
plt.imshow(train[0][0], cmap="gray")
plt.title(labels[train[0][1]])

plt.figure()
plt.imshow(train[-1][0], cmap="gray")
plt.title(labels[train[-1][1]])

# Normalization: [0, 1]
x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255
x_val = np.array(x_val) / 255

# Reshaping (5216, 150, 150) -> (5216, 150, 150, 1)
x_train = x_train.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)

# %% Data Augmentation

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(x_train)


# %% Create Deep Learning Model and Train

model = Sequential()
model.add(Conv2D(128, (7,7), strides=1, padding="same", activation="relu", input_shape=(150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))

model.add(Conv2D(64, (5,5), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))

model.add(Conv2D(32, (3,3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001)

epoch_number = 15

history = model.fit(datagen.flow(x_train, y_train, batch_size=8), 
          epochs=epoch_number, 
          validation_data=datagen.flow(x_test, y_test), 
          callbacks=[learning_rate_reduction])

print("Loss of Model: ", model.evaluate(x_test, y_test)[0])
print("Accuracy of Model: ", model.evaluate(x_test, y_test)[1] * 100)

# %% Evaluation

epochs = [i for i in range(epoch_number)]

fig, ax = plt.subplots(1,2)

train_acc = history.history["accuracy"]
train_loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
ax[0].plot(epochs, val_acc, "ro-", label="Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "go-", label="Training Loss")
ax[1].plot(epochs, val_loss, "ro-", label="Validation Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")







