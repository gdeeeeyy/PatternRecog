import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import keras as keras
from keras.layers import Dense
from keras.losses import Huber
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf

iris=load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

y_train_onehot=to_categorical(y_train)
y_test_onehot=to_categorical(y_test)


model=tf.keras.Sequential([
    Dense(units=128, activation='relu', input_dim=4),
    Dense(units=64, activation='relu'),
    Dense(units=3, activation='softmax')
])

model.compile(optimizer=Adam(), loss=Huber(), metrics=['Accuracy'])

model.fit(X_train_scaled, y_train_onehot, epochs=100, batch_size=32, verbose=1)

loss, accuracy=model.evaluate(X_test_scaled, y_test_onehot)
print(f"Test loss:{loss:.4f} \n Test accuracy:{accuracy:.4f}")