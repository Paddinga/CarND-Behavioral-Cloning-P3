import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from random import randint
import pandas as pd

def training_model():
    # Nvidia model
    model = Sequential()
    model.add(Cropping2D(cropping=((62, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, kernel_size=(5,5),strides=2, activation='relu'))
    model.add(Conv2D(36, kernel_size=(5,5),strides=2, activation='relu'))
    model.add(Conv2D(48, kernel_size=(5,5),strides=2, activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3),strides=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3),strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def read_csv(paths):
    # Load and stack driving data from different sources
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    coll_data = pd.DataFrame([])
    for path in paths:
        csv_data = pd.read_csv(path + "driving_log.csv", names = columns)
        coll_data = pd.concat([coll_data, csv_data])
    return coll_data

def prep_data(data):
    # Delete steering angles of 0° and >|2°|
    proc_data = data[data['steering'] != 0]
    proc_data = proc_data[proc_data['steering'] < 2.0]
    proc_data = proc_data[proc_data['steering'] > -2.0]
    print('Dropped steering angles >|2°| and 0°:', proc_data.shape, 'of', data.shape, 'samples left.')
    return proc_data

def preprocess(image):
    # Crop image to ROI on street
    image = aug_crop(image)
    # Normalize
    image = image / 255 - 0.5
    # Resize and return
    return cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)

def load_data(row):
    # Get steering angle
    angle = row['steering']
    # Get random
    rnd = np.random.random()
    # Pick left, center or right image and correct angle for left and right images
    if rnd < 1/3:
        image_path = row['center']
    if (rnd >= 1/3) & (rnd < 2/3):
        image_path = row['left']
        angle += 0.2
    if rnd >= 2/3:
        image_path = row['right']
        angle -= 0.2
    image = load_img(image_path)
    return image, angle

def augm_image(image, angle):
    # Get randomn for augmentation
    rnd = np.random.random(2)
    # Flipping image by chance of 50%
    if rnd[0] > 0.5:
        image, angle = aug_flip(image, angle)
    # Adjust image brightness by chance of 50%
    if rnd[1] > 0.5:
        image = aug_brightness(image)
    return image, angle

def generator(data, batch_size=64, training=True):
    images = np.zeros((batch_size, 64,64,3), dtype=np.float32)
    steer_angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        for i in range(batch_size):
            rnd = randint(0, len(data) - 1)
            row = data.iloc[rnd]
            images[i], steer_angles[i] = load_data(row)
            if training == True:
                images[i] = augm_image(images[i], steer_angles[i])
            images[i] = preprocess(images[i])
        yield images, steer_angles

# Helpers
def aug_flip(image, angle):
    return np.fliplr(image), angle*(-1)

def aug_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    adj = .25 + np.random.random()
    image[:,:,2] = image[:,:,2]* adj
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def aug_crop(image):
    top = 50
    bottom = 30
    return image[top:-bottom,:,:]

# Main pipeline
if __name__ == "__name__":
    # Paths for training data
    paths = [../data/]

    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 64

    # Prepare data
    driving_log = read_csv(paths)
    driving_log = prep_data(driving_log)
    train_samples, valid_samples = train_test_split(driving_log, test_size=0.2)

    # Use generator to augment data (in case of training)
    train_data = generator(train_samples, batch_size=BATCH_SIZE, training=True)
    valid_data = generator(valid_samples, batch_size=BATCH_SIZE, training=False)

    # Choose model
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        print('using existing model')
    else:
        model = training_model()

    # Train the model
    model.fit_generator(generator=train_data,
                        validation_data=valid_data,
                        samples_per_epoch=len(train_samples),
                        nb_epoch=EPOCHS,
                        nb_val_samples=len(valid_samples))

    # Save the model
    model.save('model.h5')