#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:00:16 2021

@author: cavid
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import cv2
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
                "emotion_dataset/train",
                target_size=(48, 48),
                batch_size=64,
                color_mode = "grayscale",
                class_mode = "categorical")

validation_generator = validation_data_gen.flow_from_directory(
                "emotion_dataset/test",
                target_size=(48,48),
                batch_size=64,
                color_mode="grayscale",
                class_mode="categorical")

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))




emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))



emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))



emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="softmax"))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7, activation="softmax"))

emotion_model.summary()


cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss = "categorical_crossentropy",
                      optimizer="Adam",
                      metrics=["accuracy"])

emotion_model_info = emotion_model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        steps_per_epoch=len(train_generator),    
        validation_steps=len(validation_generator))


model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)
    
emotion_model.save_weights("emotion_model.h5")    



















