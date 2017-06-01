import numpy as np
import cv2
from dilation_net import DilationNet
from datasets import CONFIG
from utils import interp_map
import matplotlib.pyplot as plt
import keras.backend as K


# predict function, mostly reported as it was in the original repo
def predict(image, model, ds):

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
            model_in[0] = tile  # previously tile.transpose([2, 0, 1])

            prob = model.predict(model_in)[0]

            col_prediction.append(prob)

        col_prediction = np.concatenate(col_prediction, axis=1)  # previously axis=2
        row_prediction.append(col_prediction)
    prob = np.concatenate(row_prediction, axis=0)  # previously axis=1
    if CONFIG[ds]['zoom'] > 1:
        prob = interp_map(prob, CONFIG[ds]['zoom'], image_size[1], image_size[0])

    prediction = np.argmax(prob, axis=2)  # previously axis=0
    color_image = CONFIG[ds]['palette'][prediction.ravel()].reshape(image_size)

    return color_image


if __name__ == '__main__':

    ds = 'cityscapes'  # choose between cityscapes, kitti, camvid, voc12

    # get the model
    model = DilationNet(dataset=ds, apply_softmax=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.summary()

    # read and predict a image
    im = cv2.imread(CONFIG[ds]['test_image'])
    y_img = predict(im, model, ds)

    # plot results
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(y_img)
    a.set_title('Semantic segmentation')
    plt.show(fig)

