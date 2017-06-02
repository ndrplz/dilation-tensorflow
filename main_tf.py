import tensorflow as tf
import pickle
import numpy as np
import cv2
import os.path as path
from datasets import CONFIG
from utils import interp_map


# Load dict of pretrained weights
with open('data/pretrained_dilation_cityscapes.pickle', 'rb') as f:
    w_pretrained = pickle.load(f)


def dilation_model_pretrained(input_tensor):

    def conv(name, input, strides, padding, add_bias, apply_relu, atrous_rate=None):

        with tf.variable_scope(name):

            # Load kernel weights and apply convolution
            w_kernel = w_pretrained[name + '/kernel:0']
            if not atrous_rate:
                conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
            else:
                conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
            if add_bias:
                # Load bias values and add them to conv output
                w_bias = w_pretrained[name + '/bias:0']
                conv_out = tf.nn.bias_add(conv_out, w_bias)

            if apply_relu:
                # Apply ReLu nonlinearity
                conv_out = tf.nn.relu(conv_out)

        return conv_out

    h = conv('conv1_1', input_tensor, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')

    h = conv('conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv2_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')

    h = conv('conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv3_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv3_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')

    h = conv('conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv4_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('conv4_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

    h = conv('conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
    h = conv('conv5_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
    h = conv('conv5_3', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)
    h = conv('fc6', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=4)

    h = tf.layers.dropout(h, rate=0.5, name='drop6')
    h = conv('fc7', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = tf.layers.dropout(h, rate=0.5, name='drop7')
    h = conv('final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')
    h = conv('ctx_conv1_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')
    h = conv('ctx_conv1_2', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)

    h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')
    h = conv('ctx_conv2_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=2)

    h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')
    h = conv('ctx_conv3_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=4)

    h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')
    h = conv('ctx_conv4_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=8)

    h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')
    h = conv('ctx_conv5_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=16)

    h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')
    h = conv('ctx_conv6_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=32)

    h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')
    h = conv('ctx_conv7_1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True, atrous_rate=64)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')
    h = conv('ctx_fc1', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=True)
    h = conv('ctx_final', h, strides=[1, 1, 1, 1], padding='VALID', add_bias=True, apply_relu=False)

    h = tf.image.resize_bilinear(h, size=(1024, 1024))
    logits = conv('ctx_upsample', h, strides=[1, 1, 1, 1], padding='SAME', add_bias=False, apply_relu=True)

    softmax = tf.nn.softmax(logits, dim=3, name='softmax')

    return softmax


# predict function, mostly reported as it was in the original repo
def predict(image, model, ds, sess):

    image = image.astype(np.float32) - CONFIG[ds]['mean_pixel']
    conv_margin = CONFIG[ds]['conv_margin']

    input_dims = (1,) + CONFIG[ds]['input_shape']
    batch_size, input_height, input_width, num_channels = input_dims
    model_in = np.zeros(input_dims, dtype=np.float32)

    image_size = image.shape
    output_height = input_height - 2 * conv_margin
    output_width = input_width - 2 * conv_margin
    image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
                               conv_margin, conv_margin,
                               cv2.BORDER_REFLECT_101)

    num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width  + (1 if image_size[1] % output_width else 0)

    row_prediction = []
    for h in range(num_tiles_h):
        col_prediction = []
        for w in range(num_tiles_w):
            offset = [output_height * h,
                      output_width * w]
            tile = image[offset[0]:offset[0] + input_height,
                         offset[1]:offset[1] + input_width, :]
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)

            model_in[0] = tile

            prob = sess.run(model, feed_dict={input_tensor: tile[None, ...]})[0]

            col_prediction.append(prob)

        col_prediction = np.concatenate(col_prediction, axis=1)  # previously axis=2
        row_prediction.append(col_prediction)
    prob = np.concatenate(row_prediction, axis=0)
    if CONFIG[ds]['zoom'] > 1:
        prob = interp_map(prob, CONFIG[ds]['zoom'], image_size[1], image_size[0])

    prediction = np.argmax(prob, axis=2)
    color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)

    return color_image


if __name__ == '__main__':

    input_tensor = tf.placeholder(tf.float32, shape=(None, 1396, 1396, 3))

    # Create pretrained model
    model = dilation_model_pretrained(input_tensor)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Parameters
        dataset_name = 'cityscapes'
        input_image_path  = path.join('data', dataset_name+'.png')
        output_image_path = path.join('data', dataset_name+'_out.png')

        # Read and predict on a test image
        input_image = cv2.imread(input_image_path)
        predicted_image = predict(input_image, model, dataset_name, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, predicted_image)


