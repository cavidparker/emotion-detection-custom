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

from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.models import Model

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
                "emotion_dataset/train",
                target_size=(75, 75),
                batch_size=64,
                class_mode = "categorical")

validation_generator = validation_data_gen.flow_from_directory(
                "emotion_dataset/test",
                target_size=(75,75),
                batch_size=64,
                class_mode="categorical")


emotion_model = InceptionV3(input_shape=(75,75,3), weights='imagenet', include_top=False)

for layer in emotion_model.layers:
    layer.trainable = False


from glob import glob


folders =  glob("emotion_dataset/train/*")

x = Flatten()(emotion_model.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=emotion_model.input, outputs=prediction)

model.summary()


model.compile(loss = "categorical_crossentropy",
                      optimizer="Adam",
                      metrics=["accuracy"])



emotion_model_info = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        steps_per_epoch=len(train_generator),    
        validation_steps=len(validation_generator))                      


model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)
    
emotion_model.save_weights("emotion_model.h5")



















