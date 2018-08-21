import json
import numpy as np
mapping = {
'car': 'car',
'pedestrain': 'person',
'trafficLight': 'traffic light'
}

def to_result(detection_box_list):
  bboxes = detection_box_list.get()
  clses = detection_box_list.get_field('classes')
  scores = detection_box_list.get_field('scores')
  ret = []
  #import pdb
  #pdb.set_trace()
  for i, (bbox, cls, score) in enumerate(zip(bboxes, clses, scores)):
    ret.append({
                  'id': i,
             'bbox': np.array(bbox).astype(float).tolist(),
                                'cls': cls,
                                      'score': np.array(score).tolist()
                                          })
  return ret

