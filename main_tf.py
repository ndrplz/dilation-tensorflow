import tensorflow as tf
import pickle
import cv2
import os
import os.path as path
from utils import predict
from model import dilation_model_pretrained
from datasets import CONFIG


if __name__ == '__main__':

    # Choose between 'cityscapes' and 'camvid'
    dataset = 'cityscapes'

    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Image in / out parameters
    input_image_path  = path.join('data', dataset + '.png')
    output_image_path = path.join('data', dataset + '_out.png')

    # Build pretrained model and save it as TF checkpoint
    with tf.Session() as sess:

        # Choose input shape according to dataset characteristics
        input_h, input_w, input_c = CONFIG[dataset]['input_shape']
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name('softmax:0')
        model = tf.reshape(model, shape=(1,)+CONFIG[dataset]['output_shape'])

        # Read and predict on a test image
        input_image = cv2.imread(input_image_path)
        input_tensor = graph.get_tensor_by_name('input_placeholder:0')
        predicted_image = predict(input_image, input_tensor, model, dataset, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, predicted_image)



