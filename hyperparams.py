OBSTACLE_THRESHOLD = 3
NMS_THRESHOLD = 0.5
TCONF_THRESHOLD =  0.1
STATE_THRESHOLD = 0.9
TYPE_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5

CONF_AVERAGE = 0.5

USE_HOG = False

RELIABLE_COUNT = 30

STATE_LEN = 10
TYPE_LEN = 100
TYPE_WAIT_LEN = 30


path = {
  'FCRN': 'fcrn_depth_prediction/model/NYU_FCRN.ckpt',
  'YOLO': 'keras_yolo3/model_data/yolo.h5',
  'YOLO_anchor': 'keras_yolo3/model_data/yolo_anchors.txt',
  'YOLO_classes': 'keras_yolo3/model_data/yolo.names',
}
    