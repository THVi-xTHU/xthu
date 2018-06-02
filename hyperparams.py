import numpy as np
from PIL import ImageFont

OBSTACLE_THRESHOLD = 3
NMS_TRHESHOLD = 0.5


TCONF_THREHSOLD =  0.1,
STATE_LEN = 10,
STATE_THRESHOLD = 0.9,
USE_HOG = False,

IOU_THRESHOLD = 0.5,
RELIABLE_COUNT = 5,


path = {
  'FCRN': 'fcrn_depth_prediction/model/NYU_FCRN.ckpt',
  'YOLO': 'keras_yolo3/model_data/yolo.h5',
  'YOLO_anchor': 'keras_yolo3/model_data/yolo_anchors.txt',
  'YOLO_classes': 'keras_yolo3/model_data/yolo.names',
}

efont = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
                     size=np.floor(3e-2 * 640 + 0.5).astype('int32'))
cfont = ImageFont.truetype(font='MS.ttf',
                     size=np.floor(3e-2 * 640 + 0.5).astype('int32'))
thickness = 3