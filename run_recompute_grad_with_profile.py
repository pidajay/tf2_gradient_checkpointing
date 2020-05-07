import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util import nest
from tensorflow.keras import layers, optimizers, datasets
import time
import gc
import memory_profiler
from memory_profiler import profile

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

# this is the original CNN model
def get_big_cnn_model(img_dim, n_channels, num_partitions, blocks_per_partition):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(img_dim, img_dim, n_channels)))
    for _ in range(num_partitions):
        for _ in range(blocks_per_partition):
            model.add(layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
            model.add(layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
            model.add(layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation=tf.nn.relu))
    model.add(layers.Dense(10))
    return model

def get_split_cnn_model(img_dim, n_channels, num_partitions, blocks_per_partition):
    models = [tf.keras.Sequential() for _ in range(num_partitions)]
    models[0].add(layers.Input(shape=(img_dim, img_dim, n_channels)))
    for i in range(num_partitions):
        model = models[i]
        if i > 0:
            last_shape = models[i-1].layers[-1].output_shape
            model.add(layers.Input(shape=last_shape[1:]))
        for _ in range(blocks_per_partition):
            model.add(layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
            model.add(layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
            model.add(layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu))
            model.add(layers.MaxPooling2D((1, 1), padding='same'))
    models[-1].add(layers.Flatten())
    models[-1].add(layers.Dense(32, activation=tf.nn.relu))
    models[-1].add(layers.Dense(10))
    return models

def get_data(img_dim, n_channels, batch_size):
    inputs = tf.ones([batch_size,img_dim,img_dim,n_channels])
    labels = tf.ones([batch_size], dtype=tf.int64)
    return inputs, labels

@profile
def train_no_recompute(n_steps):
    tf.random.set_seed(123)
    img_dim, n_channels, batch_size = 256, 1, 16
    x, y = get_data(img_dim, n_channels, batch_size)
    model = get_big_cnn_model(img_dim, n_channels, num_partitions=3, blocks_per_partition=3)
    optimizer = optimizers.SGD()
    losses = []
    tr_vars = model.trainable_variables
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss  = compute_loss(logits, y)
            print('loss ', loss)
            losses.append(loss)       
        grads = tape.gradient(loss, tr_vars) # tr_vars
        optimizer.apply_gradients(zip(grads, tr_vars))
        del grads 
    return losses

@profile
def train_tf_recompute(n_steps):
    tf.random.set_seed(123)
    img_dim, n_channels, batch_size = 256, 1, 16
    x, y = get_data(img_dim, n_channels, batch_size)
    models = get_split_cnn_model(img_dim, n_channels, num_partitions=3, blocks_per_partition=3)
    model1, model2, model3 = models
    model1_re = tf.recompute_grad(model1)
    model2_re = tf.recompute_grad(model2)
    model3_re = tf.recompute_grad(model3)
    optimizer = optimizers.SGD()
    tr_vars = model1.trainable_variables + model2.trainable_variables + model3.trainable_variables
    losses = []
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            logits1 = model1_re(x)
            logits2 = model2_re(logits1)
            logits3 = model3_re(logits2)
            loss  = compute_loss(logits3, y)
            print('loss ', loss)
            losses.append(loss)
        grads = tape.gradient(loss, tr_vars) # tr_vars
        optimizer.apply_gradients(zip(grads, tr_vars))
        del grads 
    return losses

start = time.time()
train_tf_recompute(1)
end = time.time()
print('Time elapsed is ', end - start, ' seconds')