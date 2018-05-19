import numpy as np
import tensorflow as tf
from . import models

class Depth(object):
    def __init__(self, model_path):
        self.input_height = 228
        self.input_width = 304
        self.input_channels = 3
        self.batch_size = 1
        self.model_path = model_path
#         tf.reset_default_graph()
        self.saver = tf.train.Saver()    
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 


    def load_model(self):

         # Create a placeholder for the input image

        self.input_node = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels))
        # Construct the network

        self.net = models.ResNet50UpProj({'data': self.input_node}, self.batch_size, 1, False)

        self.saver.restore(self.sess, self.model_path)
        
    def predict(self, img):
        img = Image.fromarray(img, 'RGB')

        img = img.resize([self.width, self.height], Image.ANTIALIAS)

        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)
        print(img.shape)
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        return pred[:, :, 0]

    def __del__(self):
        self.sess.close()
