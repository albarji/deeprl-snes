"""Custom models implementations"""

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers import Input, merge, Activation

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.model import Model


def residual_block(num_channels, layer_input, kernel_size):
    """Creates a residual block of a ResNet"""
    layer = Activation('relu')(layer_input)
    layer = Convolution2D(num_channels,  # Number convolution channels to generate
                          (kernel_size, kernel_size),  # Size of convolution kernels
                          strides=1,
                          activation='relu', padding='same')(layer)
    layer = Convolution2D(num_channels,  # Number convolution channels to generate
                          (kernel_size, kernel_size),  # Size of convolution kernels
                          strides=1, padding='same')(layer)
    res = merge.add([layer_input, layer])
    return res


def convolutional_block(num_channels, layer_input, kernel_size, pool_size):
    """Creates a convolutional block of a ResNet"""
    layer1 = Convolution2D(num_channels,  # Number convolution channels to generate
                           (kernel_size, kernel_size),  # Size of convolution kernels
                           strides=1, padding='same')(layer_input)
    layer2 = MaxPooling2D(pool_size=(pool_size, pool_size), strides=2, padding='same')(layer1)
    layer3 = residual_block(num_channels, layer2, kernel_size)
    layer4 = residual_block(num_channels, layer3, kernel_size)

    return layer4


# Create the Model Class with a deep network using keras:
class ResNet(Model):
    """Residual Network model, as used in IMPALA paper"""
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Builds and returns the output and last layer of the network."""

        kernel_size = 3  # Size of the kernel for the convolution layers
        pool_size = 3  # Size of the pooling region for the pooling layers
        image_shape = input_dict["obs"].get_shape().as_list()[1:]

        embed_input = Input(shape=image_shape, tensor=input_dict["obs"])
        layer1 = convolutional_block(16, embed_input, kernel_size, pool_size)
        layer2 = convolutional_block(32, layer1, kernel_size, pool_size)
        layer3 = convolutional_block(32, layer2, kernel_size, pool_size)

        layer4 = Flatten()(layer3)
        layer5 = Activation('relu')(layer4)
        layer5 = Dense(256, activation='relu')(layer5)
        output = Dense(num_outputs, activation=None)(layer5)

        return output, layer5


# Register models
MODELS = {
    "ResNet": ResNet
}

for key in MODELS:
    ModelCatalog.register_custom_model(key, MODELS[key])
