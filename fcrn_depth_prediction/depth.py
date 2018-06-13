import numpy as np
import tensorflow as tf
from PIL import Image
from . import models
import cv2
from hyperparams import *
class Depth(object):
    def __init__(self, model_path):
        self.input_height = 228
        self.input_width = 304
        self.input_channels = 3
        self.batch_size = 1
        self.scale = 1
        self.model_path = model_path
#         tf.reset_default_graph()
         
        
        self.load_model()


    def load_model(self):

         # Create a placeholder for the input image
        print(tf.get_default_graph())

        g=tf.Graph()
        print("g:",g)
        with g.as_default():
 
            self.input_node = tf.placeholder(tf.float32, shape=(None, self.input_height, self.input_width, self.input_channels))
            # Construct the network

            self.net = models.ResNet50UpProj({'data': self.input_node}, self.batch_size, 1, False)
            self.saver = tf.train.Saver()   
        
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            self.saver.restore(self.sess, self.model_path)
        
    def predict(self, img):
        image = Image.fromarray(img, 'RGB')

        img = image.resize([self.input_width, self.input_height], Image.ANTIALIAS)

        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        print(pred.shape)
        depth = pred[0, :, :, 0] * EXPAND
       
        depth = cv2.resize(depth, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
        print(depth.shape)
        return depth

