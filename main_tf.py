import tensorflow as tf
import numpy as np
import h5py
import pickle


weights_path = '/home/minotauro/code/dilation-tensorflow/data/pretrained_corr_channels_last.h5'
with open('data/pretrained_conv_channels_last.pickle', 'rb') as f:
    w_pretrained = pickle.load(f)


def my_constant_initializer(shape, dtype, *args, **kwargs):
    # return tf.ones(shape=shape, dtype=dtype)
    return w_pretrained['conv1_1/kernel:0']

def get_dilation_model(input_tensor, classes):

    h = tf.layers.conv2d(input_tensor, 64, (3, 3), activation=tf.nn.relu, name='conv1_1', kernel_initializer=my_constant_initializer)               # h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = tf.layers.conv2d(h, 64, (3, 3), activation=tf.nn.relu, name='conv1_2')                          # h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')     # h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = tf.layers.conv2d(h, 128, (3, 3), activation=tf.nn.relu, name='conv2_1')                         # h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = tf.layers.conv2d(h, 128, (3, 3), activation=tf.nn.relu, name='conv2_2')                         # h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')     # h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = tf.layers.conv2d(h, 256, (3, 3), activation=tf.nn.relu, name='conv3_1')                         # h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = tf.layers.conv2d(h, 256, (3, 3), activation=tf.nn.relu, name='conv3_2')                         # h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = tf.layers.conv2d(h, 256, (3, 3), activation=tf.nn.relu, name='conv3_3')                         # h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')     # h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = tf.layers.conv2d(h, 512, (3, 3), activation=tf.nn.relu, name='conv4_1')                         # h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = tf.layers.conv2d(h, 512, (3, 3), activation=tf.nn.relu, name='conv4_2')                         # h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = tf.layers.conv2d(h, 512, (3, 3), activation=tf.nn.relu, name='conv4_3')                         # h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)

    h = tf.layers.conv2d(h, 512, (3, 3), dilation_rate=(2, 2), activation=tf.nn.relu, name='conv5_1')   # h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = tf.layers.conv2d(h, 512, (3, 3), dilation_rate=(2, 2), activation=tf.nn.relu, name='conv5_2')   # h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = tf.layers.conv2d(h, 512, (3, 3), dilation_rate=(2, 2), activation=tf.nn.relu, name='conv5_3')   # h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = tf.layers.conv2d(h, 4096, (7, 7), dilation_rate=(4, 4), activation=tf.nn.relu, name='fc6')      # h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)

    h = tf.layers.dropout(h, rate=0.5, name='drop6')                                                    # h = Dropout(0.5, name='drop6')(h)
    tf.layers.conv2d(h, 4096, (1, 1), activation=tf.nn.relu, name='fc7')                                # h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = tf.layers.dropout(h, rate=0.5, name='drop7')                                                    # h = Dropout(0.5, name='drop7')(h)
    h = tf.layers.conv2d(h, classes, (1, 1), name='final')                                              # h = Convolution2D(classes, 1, 1, name='final')(h)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')                          # h = ZeroPadding2D(padding=(1, 1))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), activation=tf.nn.relu, name='ctx_conv1_1')                          # h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')                          # h = ZeroPadding2D(padding=(1, 1))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), activation=tf.nn.relu, name='ctx_conv1_2')                          # h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')                          # h = ZeroPadding2D(padding=(2, 2))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(2, 2), activation=tf.nn.relu, name='ctx_conv2_1')    # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')                          # h = ZeroPadding2D(padding=(4, 4))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(4, 4), activation=tf.nn.relu, name='ctx_conv3_1')    # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')                          # h = ZeroPadding2D(padding=(8, 8))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(8, 8), activation=tf.nn.relu, name='ctx_conv4_1')    # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')                      # h = ZeroPadding2D(padding=(16, 16))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(16, 16), activation=tf.nn.relu, name='ctx_conv5_1')  # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')                      # h = ZeroPadding2D(padding=(32, 32))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(32, 32), activation=tf.nn.relu, name='ctx_conv6_1')  # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')                      # h = ZeroPadding2D(padding=(64, 64))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), dilation_rate=(64, 64), activation=tf.nn.relu, name='ctx_conv7_1')  # h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')                         # h = ZeroPadding2D(padding=(1, 1))(h)
    h = tf.layers.conv2d(h, classes, (3, 3), activation=tf.nn.relu, name='ctx_fc1')                              # h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = tf.layers.conv2d(h, classes, (1, 1), activation=None, name='ctx_final')                                  # h = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    h = tf.image.resize_bilinear(h, size=(1024, 1024))
    logits = tf.layers.conv2d(h, classes, (16, 16), padding='same', use_bias=False, trainable=False, name='ctx_upsample')  # logits = Convolution2D(classes, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

    softmax = tf.nn.softmax(logits, dim=3, name='softmax')

    return softmax


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

    Also works reciprocally, since the transformation is its own inverse.

    # Arguments
        kernel: Numpy array (4D or 5D).

    # Returns
        The converted kernel.

    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = np.asarray(kernel)
    if not 4 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return np.copy(kernel[slices])


if __name__ == '__main__':

    input_tensor = tf.placeholder(tf.float32, shape=(None, 1396, 1396, 3))
    num_classes  = 19


    model_out = get_dilation_model(input_tensor, num_classes)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        trainable_variables = tf.trainable_variables()
        pass