OBSTACLE_THRESHOLD = 1
NMS_THRESHOLD = 0.1
TCONF_THRESHOLD =  0.3
STATE_THRESHOLD = 0.7
TYPE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.1
ASPECT_RATIO_THRESHOLD = 1
SCORE_THRESHOLD = 0.2
EVER_COLORED_THRESHOLD = 0.8
MIN_DET_COUNT = 2
MIN_HIT_THRESHOLD = 0
MAX_MISS_THRESHOLD = 20
GREEN_THRESHOLD = 5
RED_THRESHOLD = 2
CONF_AVERAGE = 0.5

EXPAND = 5

USE_HOG = False

RELIABLE_COUNT = 15

STATE_LEN = 20
TYPE_LEN = 100
TYPE_WAIT_LEN = 15
import numpy as np
from PIL import ImageFont


path = {
  'FCRN': 'fcrn_depth_prediction/model/NYU_FCRN.ckpt',
  'YOLO': 'keras_yolo3/model_data/yolo.h5',
  'YOLO_anchor': 'keras_yolo3/model_data/yolo_anchors.txt',
  'YOLO_classes': 'keras_yolo3/model_data/yolo.names',
  'KITTI_MODEL': 'KittiSeg_pretrained',
  'KITTI_RUNS': 'KittiSeg/RUNS',
  'KITTI_HYPES': 'hypes',
  'KITTI_DATA_DIR': 'KittiSeg/DATA'
}

efont = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
                     size=np.floor(3e-2 * 640 + 0.5).astype('int32'))
cfont = ImageFont.truetype(font='MS.ttf',
                     size=np.floor(3e-2 * 640 + 0.5).astype('int32'))
thickness = 5
