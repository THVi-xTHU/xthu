import json

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
  for i, (bbox, cls, score) in enumerate(zip(bboxes, clses, scores)):
    ret.append({
      'id': i,
      'bbox': bbox,
      'cls': mapping[cls],
      'score': score
    })
  return ret
