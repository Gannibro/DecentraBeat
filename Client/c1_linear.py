import flwr as fl
import tensorflow as tf
import sys
import pandas as pd
import streamlit as st

tf.random.set_seed(1)

# Learning rate
learning_rate = 0.02

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dropout(0.6),  # Dropout layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Use learning rate in Adam optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=adam_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])


