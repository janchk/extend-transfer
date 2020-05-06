from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from src.djng_app.style_transfer.utils import imread, imresize, imsave, fromimage, toimage
# from numpy import asarray as fromimage
# from imageio import imread, imsave
#
# from scipy.ndimage import imread, imresize, imsave, fromimage, toimage

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

"""
Neural Style Transfer with Keras 2.0.5

Based on:
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Contains few improvements suggested in the paper Improving the Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1605.04603).

-----------------------------------------------------------------------------------------------------------------------
"""


class Evaluator(object):
    def __init__(self):
        self.f_outputs = None
        self.loss_value = None
        self.grads_values = None
        self.eval_loss_and_grads = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x, self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


# static funcs
# -----------------------------------------------------------


def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == "channels_first":
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram


# util function to preserve image color
def original_color_transform(content, generated, mask=None):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


def load_mask(mask_path, shape, return_mask_img=False):
    if K.image_data_format() == "channels_first":
        _, channels, width, height = shape
    else:
        _, width, height, channels = shape

    mask = imread(mask_path, mode="L")  # Grayscale mask load
    mask = imresize(mask, (width, height)).astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    if return_mask_img: return mask

    mask_shape = shape[1:]

    mask_tensor = np.empty(mask_shape)

    for i in range(channels):
        if K.image_data_format() == "channels_first":
            mask_tensor[i, :, :] = mask
        else:
            mask_tensor[:, :, i] = mask

    return mask_tensor


THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


class INet:
    def __init__(self):
        self.img_size = 400
        self.pool = "Max"
        self.tv_weight = None
        self.color = "false"
        self.content_mask_path = None
        self.base_image_path = None
        self.style_image_path = None
        self.result_prefix = "res_prefix"
        self.style_masks = None
        self.content_mask = None
        self.color_mask = None
        self.image_size = 400
        self.content_weight = 0.025
        self.style_weight = [1]
        self.style_scale = 1.0
        self.total_variation_weight = 8.5e-5
        self.num_iter = 1
        self.model = "vgg19"
        self.content_loss_type = 0
        self.rescale_image = "False"
        self.rescale_method = "False"
        self.maintain_aspect_ratio = "True"
        self.content_layer = "conv5_2"
        self.init_image = "content"
        self.pool_type = "max"
        self.preserve_color = "false"
        self.min_improvement = 0.0
        self.style_weights = None
        self.pooltype = None

        self.img_width = None
        self.img_height = None
        self.img_WIDTH = None
        self.img_HEIGHT = None
        self.aspect_ratio = None

        self.evaluator = None
        self.prev_min_val = None
        self.content = None
        self.color_mask = None
        self.improvement_threshold = 0.0

        self.x = None
        self.i = 0
        self.img = None

        # self.preserve_color

    def pooling_func(self, x):
        if self.pooltype == 1:
            return AveragePooling2D((2, 2), strides=(2, 2))(x)
        else:
            return MaxPooling2D((2, 2), strides=(2, 2))(x)

    def eval_loss_and_grads(self, x, f_outputs):
        if K.image_data_format() == "channels_first":
            x = x.reshape((1, 3, self.img_width, self.img_height))
        else:
            x = x.reshape((1, self.img_width, self.img_height, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def model_construct(self, ip):
        # build the VGG16 network with our 3 images as input
        x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
        x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
        x = self.pooling_func(x)

        x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
        x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
        x = self.pooling_func(x)

        x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
        x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
        x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
        if self.model == "vgg19":
            x = Convolution2D(256, (3, 3), activation='relu', name='conv3_4', padding='same')(x)
        x = self.pooling_func(x)

        x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
        x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
        x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
        if self.model == "vgg19":
            x = Convolution2D(512, (3, 3), activation='relu', name='conv4_4', padding='same')(x)
        x = self.pooling_func(x)

        x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
        x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
        x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
        if self.model == "vgg19":
            x = Convolution2D(512, (3, 3), activation='relu', name='conv5_4', padding='same')(x)
        x = self.pooling_func(x)

        model = Model(ip, x)

        if K.image_data_format() == "channels_first":
            if self.model == "vgg19":
                weights = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', TH_19_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')
            else:
                weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')
        else:
            if self.model == "vgg19":
                weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_19_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')
            else:
                weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                                   cache_subdir='models')

        model.load_weights(weights)

        if K.backend() == 'tensorflow' and K.image_data_format() == "channels_first":
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image dimension ordering convention '
                          '(`image_dim_ordering="th"`). '
                          'For best performance, set '
                          '`image_dim_ordering="tf"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
            convert_all_kernels_in_model(model)

        print("Model loaded")

        return model

    def style_loss(self, style, combination, mask_path=None, nb_channels=None):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3

        if self.content_mask_path is not None:
            content_mask = K.variable(load_mask(self.content_mask_path, nb_channels))
            combination = combination * K.stop_gradient(content_mask)
            del content_mask

        if mask_path is not None:
            style_mask = K.variable(load_mask(mask_path, nb_channels))
            style = style * K.stop_gradient(style_mask)
            if self.content_mask_path is None:
                combination = combination * K.stop_gradient(style_mask)
            del style_mask

        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = self.img_width * self.img_height
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def content_loss(self, base, combination):
        channel_dim = 0 if K.image_data_format() == "channels_first" else -1

        try:
            channels = K.int_shape(base)[channel_dim]
        except TypeError:
            channels = K.shape(base)[channel_dim]
        size = self.img_width * self.img_height

        if self.content_loss_type == 1:
            multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
        elif self.content_loss_type == 2:
            multiplier = 1. / (channels * size)
        else:
            multiplier = 1.

        return multiplier * K.sum(K.square(combination - base))

    def total_variation_loss(self, x):
        assert K.ndim(x) == 4
        if K.image_data_format() == "channels_first":
            a = K.square(x[:, :, :self.img_width - 1, :self.img_height - 1] - x[:, :, 1:, :self.img_height - 1])
            b = K.square(x[:, :, :self.img_width - 1, :self.img_height - 1] - x[:, :, :self.img_width - 1, 1:])
        else:
            a = K.square(x[:, :self.img_width - 1, :self.img_height - 1, :] - x[:, 1:, :self.img_height - 1, :])
            b = K.square(x[:, :self.img_width - 1, :self.img_height - 1, :] - x[:, :self.img_width - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def process(self):

        base_image_path = self.base_image_path
        style_reference_image_paths = self.style_image_path

        style_image_paths = []
        for style_image_path in style_reference_image_paths:
            style_image_paths.append(style_image_path)

        style_masks_present = self.style_masks is not None
        mask_paths = []

        if style_masks_present:
            for mask_path in self.style_masks:
                mask_paths.append(mask_path)

        if style_masks_present:
            assert len(style_image_paths) == len(mask_paths), "Wrong number of style masks provided.\n" \
                                                              "Number of style images = %d, \n" \
                                                              "Number of style mask paths = %d." % \
                                                              (len(style_image_paths), len(style_masks_present))

        content_mask_present = self.content_mask is not None
        content_mask_path = self.content_mask

        color_mask_present = self.color_mask is not None

        self.rescale_image = str_to_bool(self.rescale_image)
        self.maintain_aspect_ratio = str_to_bool(self.maintain_aspect_ratio)
        self.preserve_color = str_to_bool(self.color)

        # these are the weights of the different loss components
        content_weight = self.content_weight
        total_variation_weight = self.tv_weight

        style_weights = []

        if len(style_image_paths) != len(self.style_weight):
            print("Mismatch in number of style images provided and number of style weights provided. \n"
                  "Found %d style images and %d style weights. \n"
                  "Equally distributing weights to all other styles." % (
                      len(style_image_paths), len(self.style_weight)))

            weight_sum = sum(self.style_weight) * self.style_scale
            count = len(style_image_paths)

            for i in range(len(style_image_paths)):
                style_weights.append(weight_sum / count)
        else:
            for style_weight in self.style_weight:
                style_weights.append(style_weight * self.style_scale)

        # Decide pooling function
        pooltype = str(self.pool).lower()
        assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

        self.pooltype = 1 if pooltype == "ave" else 0

        read_mode = "gray" if self.init_image == "gray" else "color"

        # dimensions of the generated picture.
        img_width = img_height = 0

        img_WIDTH = img_HEIGHT = 0
        aspect_ratio = 0

        assert self.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"

        # get tensor representations of our images
        base_image = K.variable(self.preprocess_image(base_image_path, True, read_mode=read_mode))

        style_reference_images = []
        for style_path in style_image_paths:
            style_reference_images.append(K.variable(self.preprocess_image(style_path)))

        # this will contain our generated image
        if K.image_data_format() == "channels_first":
            combination_image = K.placeholder((1, 3, self.img_width, self.img_height))
        else:
            combination_image = K.placeholder((1, self.img_width, self.img_height, 3))

        image_tensors = [base_image]
        for style_image_tensor in style_reference_images:
            image_tensors.append(style_image_tensor)
        image_tensors.append(combination_image)

        nb_tensors = len(image_tensors)
        nb_style_images = nb_tensors - 2  # Content and Output image not considered

        # combine the various images into a single Keras tensor
        input_tensor = K.concatenate(image_tensors, axis=0)

        if K.image_data_format() == "channels_first":
            shape = (nb_tensors, 3, img_width, img_height)
        else:
            shape = (nb_tensors, img_width, img_height, 3)

        ip = Input(tensor=input_tensor, batch_shape=shape)

        model = self.model_construct(ip)

        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

        if self.model == "vgg19":
            feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                              'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        else:
            feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                              'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

        # ------------------Constructing loss
        loss = K.variable(0.)
        layer_features = outputs_dict[self.content_layer]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[nb_tensors - 1, :, :, :]
        loss = loss + content_weight * self.content_loss(base_image_features, combination_features)

        nb_layers = len(feature_layers) - 1

        style_masks = []

        if style_masks_present:
            style_masks = mask_paths  # If mask present, pass dictionary of masks to style loss
        else:
            style_masks = [None for _ in range(nb_style_images)]  # If masks not present, pass None to the style loss

        channel_index = 1 if K.image_data_format() == "channels_first" else -1

        # Improvement 3 : Chained Inference without blurring
        for i in range(len(feature_layers) - 1):
            layer_features = outputs_dict[feature_layers[i]]
            shape = shape_dict[feature_layers[i]]
            combination_features = layer_features[nb_tensors - 1, :, :, :]
            style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
            sl1 = []
            for j in range(nb_style_images):
                sl1.append(self.style_loss(style_reference_features[j], combination_features, style_masks[j], shape))

            layer_features = outputs_dict[feature_layers[i + 1]]
            shape = shape_dict[feature_layers[i + 1]]
            combination_features = layer_features[nb_tensors - 1, :, :, :]
            style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
            sl2 = []
            for j in range(nb_style_images):
                sl2.append(self.style_loss(style_reference_features[j], combination_features, style_masks[j], shape))

            for j in range(nb_style_images):
                sl = sl1[j] - sl2[j]

                # Improvement 4
                # Geometric weighted scaling of style loss
                loss = loss + (style_weights[j] / (2 ** (nb_layers - (i + 1)))) * sl

        loss = loss + self.total_variation_weight * self.total_variation_loss(combination_image)

        # End loss

        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, combination_image)

        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([combination_image], outputs)
        self.evaluator = Evaluator()
        self.evaluator.f_outputs = f_outputs

        if "content" in self.init_image or "gray" in self.init_image:
            self.x = self.preprocess_image(base_image_path, True, read_mode=read_mode)
        elif "noise" in self.init_image:
            self.x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

            if K.image_data_format() == "channels_first":
                self.x = self.x.transpose((0, 3, 1, 2))
        else:
            print("Using initial image : ", self.init_image)
            self.x = self.preprocess_image(self.init_image, read_mode=read_mode)

        # We require original image if we are to preserve color in YCbCr mode
        if self.preserve_color:
            content = imread(base_image_path, mode="YCbCr")
            self.content = imresize(content, (self.img_width, self.img_height))

            if color_mask_present:
                if K.image_data_format() == "channels_first":
                    color_mask_shape = (None, None, self.img_width, self.img_height)
                else:
                    color_mask_shape = (None, self.img_width, self.img_height, None)

                self.color_mask = load_mask(self.color_mask, color_mask_shape, return_mask_img=True)
            else:
                self.color_mask = None
        else:
            self.color_mask = None

        # num_iter = self.num_iter
        self.prev_min_val = -1

        # improvement_threshold = float(self.min_improvement)

    def iterate(self):
        self.evaluator.eval_loss_and_grads = self.eval_loss_and_grads

        self.x, min_val, info = fmin_l_bfgs_b(self.evaluator.loss, self.x.flatten(),
                                              fprime=self.evaluator.grads, maxfun=20)

        if self.prev_min_val == -1:
            self.prev_min_val = min_val

        improvement = (self.prev_min_val - min_val) / self.prev_min_val * 100

        print("Current loss value:", min_val, " Improvement : %0.3f" % improvement, "%")

        self.prev_min_val = min_val

        img = self.deprocess_image(self.x.copy())

        if self.preserve_color and self.content is not None:
            img = original_color_transform(self.content, img, mask=self.color_mask)

        if not self.rescale_image:
            img_ht = int(self.img_width * self.aspect_ratio)
            print("Rescaling Image to (%d, %d)" % (self.img_width, img_ht))
            img = imresize(img, (self.img_width, img_ht), interp=self.rescale_method)

        if self.rescale_image:
            print("Rescaling Image to (%d, %d)" % (self.img_WIDTH, self.img_HEIGHT))
            img = imresize(img, (self.img_WIDTH, self.img_HEIGHT), interp=self.rescale_method)

        # fname = self.result_prefix + "_at_iteration_%d.png" % (self.i + 1)
        # imsave(fname, img)
        # end_time = time.time()
        # print("Image saved as", fname)

        self.img = img

        if self.improvement_threshold is not 0.0:
            if improvement < self.improvement_threshold and improvement is not 0.0:
                print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." %
                      (improvement, self.improvement_threshold))
                exit()

    # util function to open, resize and format pictures into appropriate tensors
    def preprocess_image(self, image_path, load_dims=False, read_mode="color"):
        # global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

        mode = "RGB" if read_mode == "color" else "L"
        img = imread(image_path, mode=mode)  # Prevents crashes due to PNG images (ARGB)

        if mode == "L":
            # Expand the 1 channel grayscale to 3 channel grayscale image
            temp = np.zeros(img.shape + (3,), dtype=np.uint8)
            temp[:, :, 0] = img
            temp[:, :, 1] = img.copy()
            temp[:, :, 2] = img.copy()

            img = temp

        if load_dims:
            self.img_WIDTH = img.shape[0]
            self.img_HEIGHT = img.shape[1]
            self.aspect_ratio = float(self.img_HEIGHT) / self.img_WIDTH

            self.img_width = self.img_size
            if self.maintain_aspect_ratio:
                self.img_height = int(self.img_width * self.aspect_ratio)
            else:
                self.img_height = self.img_size

        img = imresize(img, (self.img_width, self.img_height)).astype('float32')

        # RGB -> BGR
        img = img[:, :, ::-1]

        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

        if K.image_data_format() == "channels_first":
            img = img.transpose((2, 0, 1)).astype('float32')

        img = np.expand_dims(img, axis=0)
        return img

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        if K.image_data_format() == "channels_first":
            x = x.reshape((3, self.img_width, self.img_height))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.img_width, self.img_height, 3))

        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68

        # BGR -> RGB
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def get_result(self):
        return self.img
