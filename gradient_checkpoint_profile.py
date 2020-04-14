import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets
from gradient_checkpointing import recompute_sequential
import time

# Note - need to install python memory profiler on your machine for this to work
# pip install -U memory_profiler
# Running on CPU to measure with python memory profiler.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    


def compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

@recompute_sequential
def model_fn(model, x):
    return model(x)

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

@profile
def train():
    img_dim = 256
    n_channels = 1
    bs = 16
    x = tf.ones([bs,img_dim,img_dim,n_channels])
    y = tf.ones([bs], dtype=tf.int64)
    model = get_cnn_model(img_dim, n_channels)
    optimizer = optimizers.SGD()
    for _ in range(1):
        with tf.GradientTape() as tape: 
            logits = model_fn(model, x, num_checkpoints= 16, _watch_vars=model.trainable_variables)
            #logits = model_fn(model, x)
            loss  = compute_loss(logits, y)
            print('loss', loss) 
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del grads 
    return

# Try profiling with some inbuilt popular keras models. Currently works with vgg and mobilenet v1.
@profile
def train_popular():
    img_dim = 224 
    n_channels = 3
    bs = 64 
    x = tf.ones([bs,img_dim,img_dim,n_channels])
    y = tf.ones([bs], dtype=tf.int64)
    model = tf.keras.applications.vgg16.VGG16() #mobilenet.MobileNet() #vgg19.VGG19()
    optimizer = optimizers.SGD()
    for _ in range(1):
        with tf.GradientTape() as tape: 
            logits = model_fn(model, x, num_checkpoints=0, _watch_vars=model.trainable_variables)
            #logits = model_fn(model, x)
            loss  = compute_loss(logits, y)
            print('loss', loss) 
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del grads 
    return

start = time.time()
train() 
end = time.time()
print('Time elapsed is ', end - start, ' seconds')