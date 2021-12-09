"""
Created on Tue Nov 09 11:03:37 2021

@authors:
    Ammar Mohanna - https://www.linkedin.com/in/ammar-mohanna/
    Christian Gianoglio - https://www.linkedin.com/in/christian-gianoglio-30b514152/?originalSubdomain=it
"""
# ######################################################################################àà

#################################
# Imports and filter warnings
#################################
import os
import time
import json
import glob
import logging
import datetime
import warnings
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
from DataGenerator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet import MobileNet
from sklearn.model_selection import StratifiedKFold, train_test_split

tf.get_logger().setLevel('INFO')
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action="ignore", category=FutureWarning)  # filter warnings

#################################
# Load user configurations
#################################
with open('conf.json') as f:
    config = json.load(f)

#################################
# Configuration variables extraction
#################################
epochs = config["epochs"]
classes = config["classes"]
weights = config["weights"]
n_splits = config["n_splits"]
data_path = config["data_path"]
batch_size = config["batch_size"]
model_path = config["model_path"]
validation_split = config["validation_split"]
checkpoint_period = config["checkpoint_period"]

#################################
# Create model
#################################
if weights == "imagenet":
    base_model = MobileNet(include_top=False, weights=weights,
                             input_tensor=Input(shape=(224, 224, 3)),
                             input_shape=(224, 224, 3))
    top_layers = base_model.output
    top_layers = GlobalAveragePooling2D()(top_layers)
    top_layers = Reshape((top_layers.shape[1], 1))(top_layers)
    top_layers = AveragePooling1D(pool_size=4)(top_layers)
    top_layers = Flatten()(top_layers)
    top_layers = Dense(512, activation='relu')(top_layers)
    predictions = Dense(classes, activation='softmax')(top_layers)
    model = Model(inputs=base_model.input, outputs=predictions)
    # model.summary()
    # print ("Successfully loaded Imagenet model and weights")

#################################
# Create early stop callback
#################################
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

#################################
# Freezing all the layers except the last three
#################################
start = time.time()  # start time
# print(model.layers)
for layer in model.layers[:]:  # Freezing all layers
    layer.trainable = False
for layer in model.layers[-3:]:  # UnFreezing last three layers
    layer.trainable = True

#################################
# Compile the model
#################################
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

#################################
# Load and preprocess data
#################################
files = glob.glob(data_path + '/*/*jpg')
files = np.array(files)
samples = len(files)
print('\nNumber of samples is: ', samples, '\n')
data = np.array([])

for i in files:
    z = i.replace('\\', '/')    # solving of possible problem when running in Windows
    data = np.append(data, z)   # numpy array
# print(data)

#################################
# Data generators
#################################
labels = np.zeros(samples, dtype=int)
for i in range(samples):
    labels[i] = int(data[i].split('/')[1])  # Split and shuffle labels

skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
accuracy_history = np.array([])
loss_history = np.array([])
test_history = np.array([])
i = 1
for fu, ck in skf.split(data, labels, ):
    XTrain, XTest = data[fu], data[ck]
    yTrain, yTest = labels[fu], labels[ck]
    XVal, XTest, yVal, yTest = train_test_split(XTest, yTest, test_size=0.5, stratify=yTest, shuffle=True)

    print("Train: ", len(XTrain))
    print("Validation: ", len(XVal))
    print("Test: ", len(XTest), '\n')

    training_generator = DataGenerator(XTrain.tolist(), yTrain.tolist(), dim=(224, 224),
                                       batch_size=batch_size, n_classes=classes,
                                       n_channels=3, shuffle=True)
    validation_generator = DataGenerator(XVal.tolist(), yVal.tolist(), dim=(224, 224),
                                         batch_size=batch_size, n_classes=classes,
                                         n_channels=3, shuffle=True)
    test_generator = DataGenerator(XTest.tolist(), yTest.tolist(), dim=(224, 224),
                                   batch_size=batch_size, n_classes=classes,
                                   n_channels=3, shuffle=False)

    #################################
    # Training
    #################################
    print("Training #", i, "/", n_splits, "\n")
    history = model.fit(training_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=np.ceil(len(XVal) // batch_size),
                        steps_per_epoch=len(XTrain) // batch_size, verbose=1,
                        callbacks=early_stop)

    #################################
    # Plot training results
    #################################
    accuracy = history.history['accuracy'][-1]      # accuracy value for last epoch
    loss = history.history['loss'][-1]              # loss value for last epoch
    # print(history.history.keys())
    # # "Accuracy"
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    #################################
    # Testing model
    #################################
    prediction = model.predict(test_generator, steps=np.ceil(len(XTest) // batch_size))
    ypred = np.argmax(prediction, axis=1)
    arr = ypred - test_generator.labels[:len(ypred)]
    test_acc = np.count_nonzero(arr == 0) / len(ypred)
    print("\nTest accuracy #", i, ": ", test_acc, '\n')

    current_history  = [accuracy, loss, test_acc]
    accuracy_history = np.append(accuracy_history, current_history[0])
    loss_history     = np.append(loss_history, current_history[1])
    test_history     = np.append(test_history, current_history[2])
    i = i + 1

print("Accuracy: ", accuracy_history)
print("Loss: ", loss_history)
print("Test accuracy: ", test_history, '\n')

#################################
# Save model
#################################
print("Saving model\n")
model.save(model_path + "/save_model.h5")

#################################
# Closing
#################################
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
print("[STATUS] total duration (s): {}".format(end - start))