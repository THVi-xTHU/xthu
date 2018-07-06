from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import logging
import os
import scipy as scp
import scipy.misc
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


image_path=sys.argv[1]

sys.path.insert(1, 'KittiSeg/incl')
from KittiSeg.incl.seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import KittiSeg.incl.tensorvision.utils as tv_utils
    import KittiSeg.incl.tensorvision.core as core
except ImportError as e:
    print(e)
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


class KittiSeg(object):
    def __init__(self, default_run='KittiSeg_pretrained', runs_dir='KittiSeg/RUNS', hype_path='hypes', data_dir='KittiSeg/DATA'):
        tv_utils.set_gpus_to_use()

    # Loading hyperparameters from logdir
        self.logdir = os.path.join(runs_dir, default_run)
        os.environ["TV_DIR_DATA"] = data_dir
        self.hypes = tv_utils.load_hypes_from_logdir(self.logdir, base_path=hype_path)
        logging.info("Hypes loaded successfully.")

        self.input_height = self.hypes['jitter']['image_height']
        self.input_width = self.hypes['jitter']['image_width']
        self.input_channels = 3
        self.modules = tv_utils.load_modules_from_logdir(self.logdir)
        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        logging.info("Modules loaded successfully. Starting to build tf graph.")
        self.load_model()

    def load_model(self):

         # Create a placeholder for the input image

        # Construct the network
        g = tf.Graph()
        with g.as_default():
            self.input_node = tf.placeholder(tf.float32)
            image = tf.expand_dims(self.input_node, 0)
#             import pdb
#             pdb.set_trace()
        #self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), scope='seg')   
            self.net = core.build_inference_graph(self.hypes, self.modules,
                                                image=image)
       # self.saver = tf.train.import_meta_graph('KittiSeg/RUNS/KittiSeg_pretrained/model.ckpt-15999.meta')
            self.saver = tf.train.Saver() 
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        #self.sess = tf.InteractiveSession(graph=g)
        self.sess = tf.Session(graph=g)
        
        with self.sess.as_default():
            with g.as_default():
                #self.sess.run(tf.global_variables_initializer())
                import pdb
                pdb.set_trace()
                core.load_weights(self.logdir, self.sess, self.saver)
        
    def predict(self, input_image):
        #logging.info("Starting inference using {} as input".format(input_image))
        # Load and resize input image
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        if self.hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
            input_image = input_image.resize((self.input_width, self.input_height),
                                  Image.NEAREST)
        input_image = np.array(input_image) 
        # Run KittiSeg model on image
        feed = {self.input_node: input_image}
        softmax = self.net['softmax']
        output = self.sess.run([softmax], feed_dict=feed)

        # Reshape output from flat vector to 2D Image
        shape = input_image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(input_image, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(input_image, street_prediction)
        # Save output images to disk.
        output_base_name = 'test'

        raw_image_name = output_base_name.split('.')[0] + '_raw.png'
        rb_image_name = output_base_name.split('.')[0] + '_rb.png'
        green_image_name = output_base_name.split('.')[0] + '_green.png'

        scp.misc.imsave(raw_image_name, output_image)
        scp.misc.imsave(rb_image_name, rb_image)
        scp.misc.imsave(green_image_name, green_image)

        logging.info("")
        logging.info("Raw output image has been saved to: {}".format(
            os.path.realpath(raw_image_name)))
        logging.info("Red-Blue overlay of confs have been saved to: {}".format(
            os.path.realpath(rb_image_name)))
        logging.info("Green plot of predictions have been saved to: {}".format(
            os.path.realpath(green_image_name)))


        return output_image, rb_image


if __name__ == '__main__':
    im1 = Image.open(image_path)
    im1 = np.array(im1) 
    segmentator = KittiSeg()
    output_image, rb_image = segmentator.predict(im1)

