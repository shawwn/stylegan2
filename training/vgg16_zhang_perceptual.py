"""Tensorflow implementation of VGG perceptual distance network."""
import numpy as np
import tensorflow as tf

tft=tf.transpose
def _i(x): return tft(x,[0,2,3,1])
def _o(x): return tft(x,[0,3,1,2])

# Initialize dict which maps TF variable names to pre-trained weight dict keys.
_vgg16_params_dict = dict()

# Define ImageNet training data statistics.
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD  = [0.229, 0.224, 0.225]
ZHANG_WEIGHTING_FMAPS = [64, 128, 256, 512, 512]

#-------------------------------------------------------------------
# Load pre-trained model weights.

def get_parameters(shape, name):
    w = tf.get_variable(name, shape=shape, initializer=tf.initializers.zeros(), trainable=False, use_resource=True)
    _vgg16_params_dict[w] = name
    return w

#--------------------------------------------------------------------
# Max-pooling.

def maxpool2d(x):
    return _o(tf.nn.max_pool(_i(x), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC'))

#--------------------------------------------------------------------
# Conv. layer with bias and ReLU.

def conv2d(x, kernel, fmaps, weights_name, bias_name):
    w = get_parameters([kernel,kernel,x.shape[1].value,fmaps], weights_name)
    w = tf.cast(w, x.dtype)
    b = get_parameters([fmaps], bias_name)
    b = tf.cast(b, b.dtype)
    return tf.nn.relu(_o(tf.nn.conv2d(_i(x), w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')) + tf.reshape(b, [1, -1, 1, 1]))

#--------------------------------------------------------------------
# Zhang weighting.

def zhang_weighting(x, fmaps, weights_name):
    w = get_parameters([1,1,fmaps,1], weights_name)
    w = tf.cast(w, x.dtype)
    return _o(tf.nn.conv2d(_i(x), w, strides=[1,1,1,1], padding='SAME', data_format='NHWC'))

#----------------------------------------------------------------------------

def vgg_feature_extractor(images):
    images.set_shape([None, 3, None, None])
    x = tf.cast(images, tf.float32)

    # Normalize input to range [0, 1].
    x *=  1.0 / 255

    # Normalize according to ImageNet training data statistics.
    x = x - tf.reshape(tf.constant(VGG_MEAN, dtype=tf.float32), [1, 3, 1, 1])
    x = x * tf.reshape(tf.constant(1.0 / np.asarray(VGG_STD), dtype=tf.float32), [1, 3, 1, 1])

    # Build VGG-16 feature extractor.
    feature_channels = []

    # Conv. stack 1.
    with tf.variable_scope('conv1'):
        x = conv2d(x, 3, 64, 'conv1', 'bias1')

    with tf.variable_scope('conv2'):
        x = conv2d(x, 3, 64, 'conv2', 'bias2')

    feature_channels.append(x)
    x = maxpool2d(x)

    # Conv stack 2.
    with tf.variable_scope('conv3'):
        x = conv2d(x, 3, 128, 'conv3', 'bias3')

    with tf.variable_scope('conv4'):
        x = conv2d(x, 3, 128, 'conv4', 'bias4')

    feature_channels.append(x)
    x = maxpool2d(x)

    # Conv. stack 3.
    with tf.variable_scope('conv5'):
        x = conv2d(x, 3, 256, 'conv5', 'bias5')

    with tf.variable_scope('conv6'):
        x = conv2d(x, 3, 256, 'conv6', 'bias6')

    with tf.variable_scope('conv7'):
        x = conv2d(x, 3, 256, 'conv7', 'bias7')

    feature_channels.append(x)
    x = maxpool2d(x)

    # Conv. stack 4.
    with tf.variable_scope('conv8'):
        x = conv2d(x, 3, 512, 'conv8', 'bias8')

    with tf.variable_scope('conv9'):
        x = conv2d(x, 3, 512, 'conv9', 'bias9')

    with tf.variable_scope('conv10'):
        x = conv2d(x, 3, 512, 'conv10', 'bias10')

    feature_channels.append(x)
    x = maxpool2d(x)

    # Conv. stack 5.
    with tf.variable_scope('conv11'):
        x = conv2d(x, 3, 512, 'conv11', 'bias11')

    with tf.variable_scope('conv12'):
        x = conv2d(x, 3, 512, 'conv12', 'bias12')

    with tf.variable_scope('conv13'):
        x = conv2d(x, 3, 512, 'conv13', 'bias13')

    feature_channels.append(x)

    return feature_channels

#----------------------------------------------------------------------------

def lpips_network(images_a, images_b, **kwargs):
    """LPIPS metric using VGG-16 and Zhang weighting. (https://arxiv.org/abs/1801.03924)
    
    Takes reference images and corrupted images as an input and outputs the perceptual
    distance between the image pairs.
    """
    images_a.set_shape([None, 3, None, None])
    images_b.set_shape([None, 3, None, None])

    # Concatenate images.
    images = tf.concat([images_a, images_b], axis=0)

    # Extract features.
    vgg_features = vgg_feature_extractor(images)

    # Normalize each feature vector to unit length over channel dimension.
    normalized_features = []
    for x in vgg_features:
        n = tf.reduce_sum(x**2, axis=1, keepdims=True) ** 0.5
        normalized_features.append(x / (n + 1e-10))

    # Split and compute distances.
    diff = [tf.subtract(*tf.split(x, 2, axis=0))**2 for x in normalized_features]

    # Apply weighting from Zhang et al.
    reduced = [zhang_weighting(x, fmaps, 'perceptual_weight_%i' % i) for i, (x, fmaps) in enumerate(zip(diff, ZHANG_WEIGHTING_FMAPS))]

    # Reduce across pixels and layers.
    result = sum([tf.reduce_mean(x, axis=[2, 3]) for x in reduced])
    result = tf.reshape(result, [-1])

    return result

#----------------------------------------------------------------------------
