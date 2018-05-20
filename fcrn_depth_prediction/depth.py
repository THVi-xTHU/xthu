import numpy as np
import tensorflow as tf
from PIL import Image
from . import models

class Depth(object):
    def __init__(self, model_path):
        self.input_height = 228
        self.input_width = 304
        self.input_channels = 3
        self.batch_size = 1
        self.scale = 5
        self.model_path = model_path
#         tf.reset_default_graph()
         
        
        self.load_model()


    def load_model(self):

         # Create a placeholder for the input image

        self.input_node = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channels))
        # Construct the network

        self.net = models.ResNet50UpProj({'data': self.input_node}, self.batch_size, 1, False)
        self.saver = tf.train.Saver()   
        
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        self.saver.restore(self.sess, self.model_path)
        
    def predict(self, img):
        img = Image.fromarray(img, 'RGB')

        img = img.resize([self.input_width, self.input_height], Image.ANTIALIAS)

        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        print(pred.shape)
        return pred[0, :, :, 0] * self.scale

    def __del__(self):
        self.sess.close()
