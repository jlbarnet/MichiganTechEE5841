# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:56:28 2026

@author: jevin
"""

# === Imports ===
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
print(tf.__version__)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === Reproducibility ===
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)

# Required access method:
data = fetch_california_housing(as_frame=True)

X = data.data  # pandas DataFrame of shape (20640, 8)
y = data.target  # pandas Series

print("X shape:", X.shape)
print("y shape:", y.shape)
X.head()

# TODO: Use a 70/15/15 split (train/val/test).
# Hint: split once into train+temp, then temp into val/test.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,y,
    test_size = 0.3,
    random_state = 42
)
#need to use the left over data. Here i used before X and Y and what it did was split the total 
#and not the left over
X_val, X_test, y_val, y_test = train_test_split(
   X_temp, y_temp,
    test_size= 0.50,
    random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
# we learned in class two types of data normalization. One was the min/max and the other was std dev.
# i think this is the standard dev normalization. 
#standardization (Z-score Normalization): 
#     Xstandardized = ( X − µ)/σ 
#  

scaler = StandardScaler()

# used only on training data computes mean and standrd dev and transforms the data
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# Convert targets to float32 for TensorFlow
print("Y_TrainA:", y_train,"Y_valnB:", y_val,"Y_testB:", y_test,)
y_train_f = y_train.copy().astype("float32")
y_val_f =  y_val.copy().astype("float32")
y_test_f =  y_test.copy().astype("float32")
print("Y_TrainB:", y_train_f,"Y_valnB:", y_val_f,"Y_testB:", y_test_f)



print(X_train_s.dtype, y_train_f.dtype)


def build_single_layer_model(input_dim: int) -> keras.Model:
    """Perceptron-style: no hidden layers (linear regression)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        #one neuron, example from class had 32
        layers.Dense(1)  # linear output
        
        #our class used Activiation of Relu, if we want output to be linear, we dont include it
    ])
    return model


def build_mlp_model(input_dim: int, hidden_units=(128, 64), activation="relu") -> keras.Model:
    """MLP with at least two hidden layers."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)  # linear output for regression
    ])
    
    return model

input_dim = X_train_s.shape[1]
#this function only has one parameter
single_layer = build_single_layer_model(input_dim)
#this function has 3 parameters, 1 required  as input_dim, and two optional ones, we only pass values in if we want to
#change them. if wanted to change the non-linear term to tanh, and keep the rest i would use:
#build_mlp_model(input_dim, activation="tanh")
mlp = build_mlp_model(input_dim)

# Model Summary
single_layer.summary()
mlp.summary()

def compile_and_fit(
    model: keras.Model,
    loss_name: str,
    lr: float = 0.001,
    epochs: int =60,
    batch_size: int = 32
):
    
    model = keras.models.clone_model(model)
    model.build((None, X_train_s.shape[1]))

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr),
        loss=loss_name,
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanSquaredError(name="mse"),
        ]
    )

    history = model.fit(
        X_train_s, y_train_f,
        validation_data=(X_val_s, y_val_f),
        epochs=60,
        batch_size=32,
        verbose=1
    )
    return model, history


# TODO: Run the same MLP architecture with MSE and MAE
mlp_mse, hist_mse = compile_and_fit(mlp, loss_name="mse")
mlp_mae, hist_mae = compile_and_fit(mlp, loss_name="mae")

# Plot losses
plt.figure()
plt.plot(hist_mse.history["loss"], label="train mse-loss")
plt.plot(hist_mse.history["val_loss"], label="val mse-loss")
plt.plot(hist_mae.history["loss"], label="train mae-loss")
plt.plot(hist_mae.history["val_loss"], label="val mae-loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves: MSE vs MAE (MLP)")
plt.show()

# TODO: Train the same MLP under at least 3 learning rates (e.g., 1e-4, 1e-3, 1e-2)
lrs = [1e-4, 1e-3, 1e-2]
histories = {}

#call previous function and loop through 3 learning rates then plot
for lr in lrs:
    model_lr, hist_lr = compile_and_fit(
        mlp,
        loss_name="mse",
        lr=lr, #loop through lrs array
        
        epochs=60
    )
    histories[lr] = hist_lr

plt.figure()
for lr in lrs:
    plt.plot(histories[lr].history["val_loss"], label=f"val_loss lr={lr:g}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss (MSE)")
plt.legend()
plt.title("Convergence vs learning rate (MLP, MSE)")
plt.show()


# Utility to compute gradient norms per layer for one batch
# i didnt undertstand this section Q7
@tf.function
def batch_grad_norms(model, x_batch, y_batch, loss_fn):
    with tf.GradientTape() as tape:
        y_pred = model(__________, training=__________)
        loss = loss_fn(y_batch, y_pred)
    grads = tape.gradient(__________, model.trainable_variables)
    norms = [tf.__________(g) for g in grads if g is not None]
    return loss, norms


# Prepare a small batch
x_b = tf.convert_to_tensor(X_train_s[:512], dtype=tf.__________)
y_b = tf.convert_to_tensor(y_train_f[:512], dtype=tf.__________)


# Use a freshly compiled model (MSE)
probe_model = keras.models.__________(mlp)
probe_model.build((__________, X_train_s.shape[__________]))
loss_fn = keras.losses.__________()


# Initialize weights by running one forward pass
_ = probe_model(__________)


loss_val, norms = batch_grad_norms(probe_model, x_b, y_b, loss_fn)

print("Batch loss:", float(loss_val))
print("Num trainable vars:", len(probe_model.trainable_variables))
print("First 10 grad norms:", [float(n) for n in norms[:10]])


def build_deep_mlp(input_dim: int, depth: int, width: int = 128, activation="relu") -> keras.Model:
    layers_list = [layers.Input(shape=(input_dim,))]
    for _ in range(depth):
        layers_list.append(layers.Dense(width, activation="relu"))
    layers_list.append(layers.Dense(1))
    return keras.Sequential(layers_list)

depths = [1, 3, 5]
depth_hist = {}

for d in depths:
    model_d = build_deep_mlp(input_dim, depth=d, width=128, activation="relu")
    model_d, hist_d = compile_and_fit(model_d, loss_name="mse", lr=1e-3, epochs=60)
    depth_hist[d] = hist_d

plt.figure()
for d in depths:
    plt.plot(depth_hist[d].history["val_loss"], label=f"depth={d}")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss (MSE)")
plt.legend()
plt.title("Validation loss vs depth (MLP)")
plt.show()

def evaluate(model: keras.Model, X, y):
    return dict(zip(model.metrics_names, model.evaluate(X, y, verbose=0)))

# Choose one trained model as your final (TODO: replace with your best model)
final_model = model_d

results_test = evaluate(final_model, X_test_s, y_test_f)#use float test for tf
results_val  = evaluate(final_model, X_val_s, y_val_f)#use float test for tf

print("Validation:", results_val)
print("Test:", results_test)