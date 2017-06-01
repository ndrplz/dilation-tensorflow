import warnings

from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Input, AtrousConvolution2D
from keras.layers import Dropout, UpSampling2D, ZeroPadding2D

from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

from datasets import CONFIG
from utils import softmax


# CITYSCAPES MODEL
def get_dilation_model_cityscapes(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    h = UpSampling2D(size=(8, 8))(h)
    logits = Convolution2D(classes, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_cityscapes')

    return model


# PASCAL VOC MODEL
def get_dilation_model_voc(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, activation='relu', name='fc-final')(h)
    h = ZeroPadding2D(padding=(33, 33))(h)
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_1')(h)
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_2')(h)
    h = AtrousConvolution2D(4 * classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ct_conv2_1')(h)
    h = AtrousConvolution2D(8 * classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ct_conv3_1')(h)
    h = AtrousConvolution2D(16 * classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ct_conv4_1')(h)
    h = AtrousConvolution2D(32 * classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ct_conv5_1')(h)
    h = Convolution2D(32 * classes, 3, 3, activation='relu', name='ct_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ct_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_voc12')

    return model


# KITTI MODEL
def get_dilation_model_kitti(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_kitti')

    return model


# CAMVID MODEL
def get_dilation_model_camvid(input_shape, apply_softmax, input_tensor, classes):

    if input_tensor is None:
        model_in = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            model_in = Input(tensor=input_tensor, shape=input_shape)
        else:
            model_in = input_tensor

    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    logits = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out, name='dilation_camvid')

    return model


# model function
def DilationNet(dataset, input_shape=None, apply_softmax=True, pretrained=True,
                input_tensor=None, classes=None):
    """ Instantiate the Dilation network architecture, optionally loading weights
        pre-trained on a dataset in the set (cityscapes, voc12, kitti, camvid).
        Note that pre-trained model is only available for Theano dim ordering.

        The model and the weights should be compatible with both
        TensorFlow and Theano backends.

        # Arguments
            dataset: choose among (cityscapes, voc12, kitti, camvid).
            input_shape: shape tuple. It should have exactly 3 inputs channels,
                and the axis ordering should be coherent with what specified in
                your keras.json (e.g. use (3, 512, 512) for 'th' and (512, 512, 3)
                for 'tf'). None will default to dataset specific sizes.
            apply_softmax: whether to apply softmax or return logits.
            pretrained: boolean. If `True`, loads weights coherently with
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of segmentation classes. If pretrained is True,
                it should be coherent with the dataset chosen.

        # Returns
            A Keras model instance.
    """
    if dataset not in {'cityscapes', 'voc12', 'kitti', 'camvid'}:
        raise ValueError('The `dataset` argument should be one among '
                         '(cityscapes, voc12, kitti, camvid)')

    if classes is not None:
        if classes != CONFIG[dataset]['classes'] and pretrained:
            raise ValueError('Cannot load pretrained model for dataset `{}` '
                             'with {} classes'.format(dataset, classes))
    else:
        classes = CONFIG[dataset]['classes']

    if input_shape is None:
        input_shape = CONFIG[dataset]['input_shape']

    # get the model
    if dataset == 'cityscapes':
        model = get_dilation_model_cityscapes(input_shape=input_shape, apply_softmax=apply_softmax,
                                              input_tensor=input_tensor, classes=classes)
    elif dataset == 'voc12':
        model = get_dilation_model_voc(input_shape=input_shape, apply_softmax=apply_softmax,
                                       input_tensor=input_tensor, classes=classes)
    elif dataset == 'kitti':
        model = get_dilation_model_kitti(input_shape=input_shape, apply_softmax=apply_softmax,
                                         input_tensor=input_tensor, classes=classes)
    elif dataset == 'camvid':
        model = get_dilation_model_camvid(input_shape=input_shape, apply_softmax=apply_softmax,
                                          input_tensor=input_tensor, classes=classes)

    # load weights
    if pretrained:

        weights_path = get_file(CONFIG[dataset]['weights_file'],
                                CONFIG[dataset]['weights_url'],
                                cache_subdir='models')
        model.load_weights(weights_path)
        convert_all_kernels_in_model(model)


        # if K.image_dim_ordering() == 'th':
        #     weights_path = get_file(CONFIG[dataset]['weights_file'],
        #                             CONFIG[dataset]['weights_url'],
        #                             cache_subdir='models')
        #
        #     model.load_weights(weights_path)
        #
        #     if K.backend() == 'tensorflow':
        #         warnings.warn('You are using the TensorFlow backend, yet you '
        #                       'are using the Theano '
        #                       'image dimension ordering convention '
        #                       '(`image_dim_ordering="th"`). '
        #                       'For best performance, set '
        #                       '`image_dim_ordering="tf"` in '
        #                       'your Keras config '
        #                       'at ~/.keras/keras.json.')
        #         convert_all_kernels_in_model(model)
        # else:
        #     convert_all_kernels_in_model(model)
        #     raise NotImplementedError('Pretrained DilationNet model is not available with tensorflow dim ordering')

    return model
