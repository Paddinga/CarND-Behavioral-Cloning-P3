import os
import csv
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import pandas as pd
import math
from random import randint

def training_model():
    # Nvidia model
    model = Sequential()
    model.add(Cropping2D(cropping=((62, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, 5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5,5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3,3, subsample=(1,1), activation='relu'))
    model.add(Conv2D(64, 3,3, subsample=(1,1), activation='relu'))
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

def prepare_data(paths):
    image_paths = []
    angles = []
    for i in range(len(paths)):
        with open(paths[i] + "driving_log.csv", newline='') as csv_data:
            driving_data = list(csv.reader(csv_data, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
        for row in driving_data[1:]:
            # Drop data with speed < 0.1
            if float(row[6]) < 0.1:
                continue
            # Drop data with steering angle = 0
            if float(row[3]) == 0:
                continue
            # Drop data with absolute steering angle > 3.0
            if (float(row[3]) > 3.0) | (float(row[3]) < - 3.0):
                continue
            image_paths.append(row[0])
            angles.append(float(row[3]))
            # get left image path and angle
            image_paths.append(row[1])
            angles.append(float(row[3]) + 0.2)
            # get left image path and angle
            image_paths.append(row[2])
            angles.append(float(row[3]) - 0.2)
    image_paths = np.array(image_paths)
    angles = np.array(angles)
    print('Data prepared', image_paths.shape, angles.shape)
    return image_paths, angles


def preprocess(image):
    # Crop image to ROI on street
    img = aug_crop(image)
    # Normalize
    img = img / 255. - 0.5
    # Resize and return
    return cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)

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


def load_image(row, psc):
    angle = np.float(row['steer'])
    rnd = np.random.random()
    if rnd < 1 / 3:
        view = 'center'
    if (rnd >= 1 / 3) & (rnd < 2 / 3):
        view = 'left'
        angle += 0.2
    if rnd >= 2 / 3:
        view = 'right'
        angle -= 0.2
    image_src = str(row[view])
    image_src = image_src.split('/')[-1]
    image_src = os.path.join('./data_rec/IMG/', image_src)
    if os.path.exists(image_src):
        image = ndimage.imread(image_src)
        psc += 1
    else:
        print(image_src, psc)
    return image, angle

def generator(driving_log, psc,  batch_size=64, training=True):
    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    # Running a loop
    while True:
        for i in range(batch_size):
            rnd_row = randint(0,len(driving_log)-1)
            row = driving_log.iloc[rnd_row]
            image, angle = load_image(row, psc)
            if training == True:
                image, angle = augm_image(image, angle)
            #image = preprocess(image)
            images[i] = image
            angles[i] = angle
        yield images, angles

# Helpers
def aug_flip(image, angle):
    return np.fliplr(image), angle*(-1)

def aug_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    adj = .25 + np.random.random()
    image[:,:,2] = image[:,:,2]* adj
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def aug_crop(image):
    top = 50
    bottom = 30
    return image[top:-bottom,:,:]

# Main pipeline
if __name__ == "__main__":
    # Path for csv
    csv_path = 'data_rec/driving_log_pro.csv'
    # Read csv into driving_log
    driving_log = pd.read_csv(csv_path, names=['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed'])
    # Shuffle data in driving log
    driving_log = driving_log[1:]
    driving_log = driving_log.sample(frac=1).reset_index(drop=True)
    #Split into train and validation sets
    row_split = math.floor(driving_log.shape[0] * 0.8)
    train_data = driving_log.loc[0:row_split]
    valid_data = driving_log.loc[row_split+1:]
    # Hyperparameters
    EPOCHS = 19
    BATCH_SIZE = 128
    psc = 1
    # Use generator to augment data (in case of training)
    print('# Generating data...')
    gen_train = generator(train_data, psc,  batch_size=BATCH_SIZE, training=True)
    gen_valid = generator(valid_data, psc, batch_size=BATCH_SIZE, training=False)
    # Choose model
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        print('using existing model')
    else:
        model = training_model()

    # Train the model
    model.fit_generator(generator=gen_train,
                        validation_data=gen_valid,
                        samples_per_epoch=len(train_data),
                        nb_epoch=EPOCHS,
                        nb_val_samples=len(valid_data),
                        verbose=2)

    # Save the model
    model.save('model.h5')
    print('# Model saved.')