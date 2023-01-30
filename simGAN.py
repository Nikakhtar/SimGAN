import sys
import time
import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from itertools import groupby
from skimage.util import montage

from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate, Activation, Add
from keras.layers.convolutional import UpSampling2D, MaxPooling2D, Conv2DTranspose, Conv2D
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras import initializers
from keras import applications
from keras.utils import plot_model
from keras.preprocessing import image

import tensorflow as tf
###############################
img_width = 55
img_height = 35
channels = 1

batch_size = 512
###############################
def self_regularisation_loss(y_true, y_pred):
    return tf.multiply(0.0002, tf.reduce_sum(tf.abs(y_pred - y_true)))

# reduce_sum: Computes the sum of elements across dimensions of a tensor.
###############################
def local_adversarial_loss(y_true, y_pred):
    truth = tf.reshape(y_true, (-1, 2))
    predicted = tf.reshape(y_pred, (-1, 2))
    
    computed_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth, logits=predicted)
    output = tf.reduce_mean(computed_loss)
    
    return output
################################
def refiner_model(width = 55, height = 35, channels = 1):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.
    
    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    
    def resnet_block(input_features, nb_features=64, kernel_size=3):
        """
        A ResNet block with two `kernel_size` x `kernel_size` convolutional layers,
        each with `nb_features` feature maps.
        
        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.
        
        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(input_features)
        y = Activation('relu')(y)
        y = Conv2D(nb_features, kernel_size=kernel_size, padding='same')(y)
        
        y = Add()([y, input_features])
        y = Activation('relu')(y)
        
        return y

    input_layer = Input(shape=(height, width, channels))
    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)

    for _ in range(4):
        x = resnet_block(x)

    output_layer = Conv2D(channels, kernel_size=1, padding='same', activation='tanh')(x)

    return Model(input_layer, output_layer, name='refiner')
##################################
def discriminator_model(width = 55, height = 35, channels = 1):
    input_layer = Input(shape=(height, width, channels))

    x = Conv2D(96, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(2, kernel_size=1, strides=1, padding='same', activation='relu')(x)
    output_layer = Reshape(target_shape=(-1, 2))(x)

    return Model(input_layer, output_layer, name='discriminator')
#################################
refiner = refiner_model(img_width, img_height, channels)
refiner.compile(loss=self_regularisation_loss, optimizer=SGD(lr=0.001))

refiner.summary()
################################
disc = discriminator_model(img_width, img_height, channels)
disc.compile(loss=local_adversarial_loss, optimizer=SGD(lr=0.001))

disc.summary()
###############################
synthetic_img = Input(shape=(img_height, img_width, channels))
refined_output = refiner(synthetic_img)
discriminator_output = disc(refined_output)

combined_model = Model(inputs=synthetic_img, outputs=[refined_output, discriminator_output], name='combined')
combined_model.summary()
################################
disc.trainabler = False
combined_model.compile(loss=[self_regularisation_loss, local_adversarial_loss], optimizer=SGD(lr=0.001))
#################################
path = os.path.dirname(os.path.abspath('.'))
data_dir = os.path.join('.', 'input')
cache_dir = '.'
################################
# load the data file and extract dimentions
with h5py.File(os.path.join(data_dir, 'gaze.h5'), 'r') as t_file:
    syn_img_stack = np.stack([np.expand_dims(a, -1) for a in t_file['image'].values()], 0)
    
with h5py.File(os.path.join(data_dir, 'real_gaze.h5'), 'r') as t_file:
    real_img_stack = np.stack([np.expand_dims(a, -1) for a in t_file['image'].values()], 0)
#################################
"""
Module to plot a batch of images along w/ their corresponding label(s)/annotations and save the plot to disc/show them.

Use cases:
Plot images along w/ their corresponding ground-truth label & model predicted label,
Plot images generated by a GAN along w/ any annotations used to generate these images,
Plot synthetic, generated, refined, and real images and see how they compare as training progresses in a GAN,
etc...
"""
plotted_imgs = 16

def plot_batch(image_batch, figure_path, label_batch=None):
    
    all_groups = {label: montage(np.stack([img[:,:,0] for img, lab in img_lab_list],0)) 
                  for label, img_lab_list in groupby(zip(image_batch, label_batch), lambda x: x[1])}
    fig, c_axs = plt.subplots(1,len(all_groups), figsize=(len(all_groups)*4, 8), dpi = 600)
    for c_ax, (c_label, c_mtg) in zip(c_axs, all_groups.items()):
        c_ax.imshow(c_mtg, cmap='bone')
        c_ax.set_title(c_label)
        c_ax.axis('off')
    # fig.savefig(os.path.join(figure_path))
    plt.show()
    plt.close()
##################################
"""
Module implementing the image history buffer described in `2.3. Updating Discriminator using a History of
Refined Images` of https://arxiv.org/pdf/1612.07828v1.pdf.
"""

class ImageHistoryBuffer():
    def __init__(self, shape, max_size, batch_size):
        """
        :param shape: Shape of the data to be stored in the image history buffer
                      (i.e. (0, img_height, img_width, img_channels)).
        :param max_size: Maximum number of images that can be stored in the image history buffer.
        :param batch_size: Batch size used to train GAN.
        """
        self.image_history_buffer = np.zeros(shape=shape)
        self.max_size = max_size
        self.batch_size = batch_size
        
    def add_to_history_img_buffer(self, images, nb_to_add=0):
        if not nb_to_add:
            nb_to_add = self.batch_size // 2
        
        if len(self.image_history_buffer) < self.max_size:
            np.append(self.image_history_buffer, images[:nb_to_add], axis=0)
        elif len(self.image_history_buffer) == self.max_size:
            self.image_history_buffer[:nb_to_add] = images[:nb_to_add]
        else:
            assert False

        np.random.shuffle(self.image_history_buffer)
    
    def get_from_image_history_buffer(self, nb_to_get=None):
        """
        Get a random sample of images from the history buffer.

        :param nb_to_get: Number of images to get from the image history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` images from the image history buffer, or an empty np array if the image
                 history buffer is empty.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        try:
            return self.image_history_buffer[:nb_to_get]
        except IndexError:
            return np.zeros(shape=0)
###################################
datagen = image.ImageDataGenerator(preprocessing_function=applications.xception.preprocess_input, data_format='channels_last')
##################################
syn_gen = datagen.flow(x=syn_img_stack, batch_size=batch_size)
real_gen = datagen.flow(x=real_img_stack, batch_size=batch_size)
##################################
disc_output_shape = disc.output_shape
##################################
def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch
##################################
def pretrain_gen(steps, log_interval, save_path, profiling=True):
    losses = []
    gen_loss = 0.
    if profiling:
        start = time.perf_counter()
    for i in range(steps):
        syn_imgs_batch = get_image_batch(syn_gen)
        loss = refiner.train_on_batch(syn_imgs_batch, syn_imgs_batch)
        gen_loss += loss

        if (i+1) % log_interval == 0:
            print('pre-training generator step {}/{}: loss = {:.5f}'.format(i+1, steps, gen_loss / log_interval))
            losses.append(gen_loss / log_interval)
            gen_loss = 0.
        
        if (i+1) % (5*log_interval) == 0:
            figure_name = 'refined_img_pretrain_step_{}.png'.format(i)
            syn_imgs = get_image_batch(syn_gen)[:plotted_imgs]
            gen_imgs = refiner.predict_on_batch(syn_imgs)

            plot_batch(np.concatenate((syn_imgs, gen_imgs)), os.path.join(cache_dir, figure_name), 
                       label_batch=['Synthetic'] * plotted_imgs + ['Refined'] * plotted_imgs)

    if profiling:
        duration = time.perf_counter() - start
        print('pre-training the refiner model for {} steps lasted = {:.2f} minutes = {:.2f} hours'.format(steps, duration/60., duration/3600.))
    
#     refiner.save(save_path)
    
    return losses
######################################
# we first train the Rθ network with just self-regularization loss for 1,000 steps
gen_pre_steps = 1000
gen_log_interval = 20

pre_gen_path = os.path.join(cache_dir, 'refiner_model_pre_trained_{}.h5'.format(gen_pre_steps))
if os.path.isfile(pre_gen_path):
    refiner.load_weights(pre_gen_path)
    print('loading pretrained model weights')
else:
    losses = pretrain_gen(gen_pre_steps, gen_log_interval, pre_gen_path)
    plt.plot(range(gen_log_interval, gen_pre_steps+1, gen_log_interval), losses)
