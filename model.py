import os
import cv2
import numpy as np
from scipy import ndimage
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
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


def load_image(row):
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
    image_src = os.path.join('./data/IMG/', image_src)
    image = ndimage.imread(image_src)
    return image, angle

def generator(driving_log,  batch_size=64, training=True):
    images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    # Running a loop
    while True:
        for i in range(batch_size):
            rnd_row = randint(0,len(driving_log)-1)
            row = driving_log.iloc[rnd_row]
            image, angle = load_image(row)
            if training == True:
                image, angle = augm_image(image, angle)
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

# Main pipeline
if __name__ == "__main__":
    # Path for csv
    csv_path = 'data/driving_log.csv'
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
    # Use generator to augment data (in case of training)
    print('# Generating data...')
    gen_train = generator(train_data,  batch_size=BATCH_SIZE, training=True)
    gen_valid = generator(valid_data, batch_size=BATCH_SIZE, training=False)
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