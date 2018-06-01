obstacle_threshold=3

config = {}

config['LightTracker'] = {
  'conf_threshold': 0.1,
  'state_len': 10,
  'state_threshold': 0.9,
  'use_hog': False,
}

config['LightPool'] = {
  'conf_threshold': 0.1, 
  'iou_threshold': 0.5, 
  'reliable_count': 5,
  'use_hog': True,
}




path = {
  'FCRN': 'fcrn_depth_prediction/model/NYU_FCRN.ckpt',
  'YOLO': 'keras_yolo3/model_data/yolo.h5',
  'YOLO_anchor': 'keras_yolo3/model_data/yolo_anchors.txt',
  'YOLO_classes': 'keras_yolo3/model_data/yolo.names',
}
    