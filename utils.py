import cv2
import numpy as np
from datasets import CONFIG
#import numba


# this function is the same as the one in the original repository
# basically it performs upsampling for datasets having zoom > 1
# @numba.jit(nopython=True)
def interp_map(prob, zoom, width, height):
    channels = prob.shape[2]
    zoom_prob = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[r1, c0, c] + (1 - rt) * prob[r0, c0, c]
                v1 = rt * prob[r1, c1, c] + (1 - rt) * prob[r0, c1, c]
                zoom_prob[h, w, c] = (1 - ct) * v0 + ct * v1
    return zoom_prob


# predict function, mostly reported as it was in the original repo
def predict(image, input_tensor, model, ds, sess):

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