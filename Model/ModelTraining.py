#!/usr/bin/env python

# METADATA

__author__ = "Adrian Rodriguez-Bazaga, Josep M. Porta"
__credits__ = ["Adrian Rodriguez-Bazaga", "Josep M. Porta"]
__version__ = "1.0.0"
__email__ = "adrianrodriguezbazaga@gmail.com"

# PACKAGES

# Basic system routines packages
import os, sys

# Keras imports
from keras import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras import backend as K
from keras.models import *

# Others
from pandas import *
import argparse

# Add the Utils folder path to the sys.path list
sys.path.append('../Tools/')

# Import model custom metrics
from CustomModelMetrics import m_recall
from CustomModelMetrics import m_f1
from CustomModelMetrics import m_precision

def usage():
    print("Usage Examples:")
    print("python ModelTraining.py --help")
    print("python ModelTraining.py --trainDir='trainDirPath' --validationDir='validationDirPath'")
    print("python ModelTraining.py --trainDir='trainDirPath' --validationDir='validationDirPath' --epochs=300 --batch_size=32")
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description = "Script to train the CNN model", epilog = "")
    parser.add_argument("-td", "--trainDir", required=True, help="Path to the directory containing the training patches (REQUIRED).", dest="trainDir")
    parser.add_argument("-vd", "--validationDir", required=True, help="Path to the directory containing the validation patches (REQUIRED).", dest="validationDir")
    parser.add_argument("-e", "--epochs", default=300, required=False, help="Number of training epochs (OPTIONAL).", dest="epochs")
    parser.add_argument("-bs", "--batch_size", default=32, required=False, help="Training batch size (OPTIONAL).", dest="batch_size")
    parser.add_argument("-u", "--usage", help="Usage examples", dest="usage", action = 'store_true')
    return parser.parse_args()

def create_model(img_width, img_height):
    # Use the correct Keras input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', m_recall, m_precision, m_f1])
    
    return model

# SCRIPT
if __name__ == '__main__':
    args = parse_args()
    if(args.usage):
        usage()

    # Run options and parameters
    train_data_dir = args.trainDir
    validation_data_dir = args.validationDir
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)

    # Our images dimension
    img_width, img_height = 64, 64

    # Build the model
    model = create_model(img_width, img_height)

    # Augmentation configuration for the training and validation sets + rescaling
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Train and validation datagens
    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size)
    validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size)

    # Save best weights found so far (checkpoint) and early stopping
    bestWeightsPath = "Model_Weights.h5"
    checkpoint = callbacks.ModelCheckpoint(bestWeightsPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stop = callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max') 
    callbacks_list = [checkpoint, early_stop]

    # Save the model training history as CSV
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.n//validation_generator.batch_size, callbacks=callbacks_list)
    pandas.DataFrame(history.history).to_csv("ModelTraining-History.csv")

    # Serialize model to JSON
    model_json = model.to_json()
    with open("ModelTraining-JSON.json", "w") as json_file:
        json_file.write(model_json)
