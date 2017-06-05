import tensorflow as tf
import pickle
import cv2
import os.path as path
from utils import predict
from model import dilation_model_pretrained
from datasets import CONFIG


if __name__ == '__main__':

    # Choose between 'cityscapes' and 'camvid'
    dataset = 'camvid'

    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Choose input shape according to dataset characteristics
    input_h, input_w, input_c = CONFIG[dataset]['input_shape']
    input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c))

    # Create pretrained model
    model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

    # Test pretrained model
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Parameters
        input_image_path  = path.join('data', dataset+'.png')
        output_image_path = path.join('data', dataset+'_out.png')

        # Read and predict on a test image
        input_image = cv2.imread(input_image_path)
        predicted_image = predict(input_image, input_tensor, model, dataset, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, predicted_image)


