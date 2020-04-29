import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util import nest
from tensorflow.keras import layers, optimizers, datasets
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

# this is the original CNN model
def get_cnn_model(img_dim, n_channels):
    model = tf.keras.Sequential([
    layers.Reshape(
        target_shape=[img_dim, img_dim, n_channels],
        input_shape=(img_dim, img_dim, n_channels)),
    layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(60, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
    return model

# The following blocks are the above CNN model that has been partitioned into 3 blocks
def model_block1(img_dim, n_channels):
    model = tf.keras.Sequential([
    layers.Reshape(
        target_shape=[img_dim, img_dim, n_channels],
        input_shape=(img_dim, img_dim, n_channels)),
    layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same')])
    return model

def model_block2():
    model = tf.keras.Sequential([
    layers.Input(shape=(256,256,20)),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(60, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same')])
    return model

def model_block3():
    model = tf.keras.Sequential([
    layers.Input(shape=(256,256,40)),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])
    return model


@profile
def train_no_recompute():
    img_dim = 256
    n_channels = 1
    bs = 16
    x = tf.ones([bs,img_dim,img_dim,n_channels])
    y = tf.ones([bs], dtype=tf.int64)
    model = get_cnn_model(img_dim, n_channels)
    optimizer = optimizers.SGD()
    tr_vars = model.trainable_variables
    for _ in range(1):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss  = compute_loss(logits, y)
            print('loss', loss)
        grads = tape.gradient(loss, tr_vars) # tr_vars
        optimizer.apply_gradients(zip(grads, tr_vars))
        del grads 
    return

@profile
def train_tf_recompute():
    img_dim = 256
    n_channels = 1
    bs = 16
    x = tf.ones([bs,img_dim,img_dim,n_channels])
    y = tf.ones([bs], dtype=tf.int64)
    bk1_orig = model_block1(img_dim, n_channels)
    bk1 = tf.recompute_grad(bk1_orig)
    bk2_orig =  model_block2()
    bk2 = tf.recompute_grad(bk2_orig)
    bk3_orig = model_block3()
    bk3 = tf.recompute_grad(bk3_orig)
    optimizer = optimizers.SGD()
    tr_vars = bk1_orig.trainable_variables + bk2_orig.trainable_variables + bk3_orig.trainable_variables
    for _ in range(1):
        with tf.GradientTape() as tape:
            logits1 = bk1(x, trainable_variables=bk1_orig.trainable_variables)
            logits2 = bk2(logits1, trainable_variables=bk2_orig.trainable_variables)
            logits3 = bk3(logits2, trainable_variables=bk3_orig.trainable_variables)
            loss  = compute_loss(logits3, y)
            print('loss', loss)
        grads = tape.gradient(loss, tr_vars) # tr_vars
        optimizer.apply_gradients(zip(grads, tr_vars))
        del grads 

start = time.time()
train_tf_recompute()
end = time.time()
print('Time elapsed is ', end - start, ' seconds')