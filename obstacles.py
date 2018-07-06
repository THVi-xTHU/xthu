# This class implements the functionality of obstacle tracking and trajectory prediction, 
# when obstacles forwarding towards the blind and the predicted distance is lower than a threshold,
# the system will force the blind to stop and wait
#
#

from __future__ import print_function

from numba import jit
from KCF.kcftracker import KCFTracker
import os.path
import numpy as np
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from util.box_list import *
from util.box_list_ops import *
from non_maximum_suppression import non_max_suppression_slow
from hyperparams import *
import pdb
@jit
def iou_elementwise(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox, bbox_depth=None):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  if bbox_depth is not None:
    d = np.mean(bbox_depth)
  else:
    d = np.nan
  return np.array([x,y,d,s,r]).reshape((5,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[3]*x[4])
  h = x[3]/w
  if(score==None):
    return (np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((4,)), x[2])
  else:
    return (np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((5,)), x[2])

with open('keras_yolo3/model_data/yolo.names', 'r') as fr:
  all_classes = fr.read().strip().split('\n')

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = np.zeros((len(all_classes),))
  def __init__(self, box, cls, depth, image, use_hog=True):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=9, dim_z=5)
    #x = [u, v, h, s, r, du, dv, dh, ds]
    self.kf.F = np.array([[1,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,1,0], [0,0,0,1,0,0,0,0,1], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,1,0,0,0], 
                         [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]]) # state transition matrix
    # y = [u,v,h,s,r]
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]]) # Measurement function

    self.kf.R[3:,3:] *= 10.   # state uncertainty
    self.kf.P[5:,5:] *= 1000. # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.          # covariance matrix
    self.kf.Q[-1,-1] *= 0.01  # process uncertainty
    self.kf.Q[5:,5:] *= 0.01  

    self.kf.x[:5] = convert_bbox_to_z(box, depth) # initial state (location and velocity)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count[all_classes.index(cls)]
    KalmanBoxTracker.count[all_classes.index(cls)] += 1
    print('Class %s: %d'%(cls, KalmanBoxTracker.count[all_classes.index(cls)]))
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.cls = cls 
    self.state = 'd'
    self.tracker = KCFTracker(use_hog, True, True)
    self.bbox = two_point2point_size(box)
    z = convert_bbox_to_z(box, depth)
    self.history = [(box, z[2])]
    self.tracker.init(self.bbox, image)
    print('[Class %s][Tracker %d]initial box:'%(self.cls, self.id), box)
    

  def update(self, bbox, cls, depth):
    """
    Updates the state vector with observed bbox.
    """
    assert cls == self.cls, 'matched to unconsistent class: before: %s, after %s'%(self.cls, cls)
    self.time_since_update = 0
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox, depth))

  def predict(self, image):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[8]+self.kf.x[3])<=0):
      self.kf.x[8] *= 0.0
    print(self.tracker._roi)
    pbox, peak_value = self.tracker.predict(image)
    if peak_value >= TCONF_THRESHOLD:
      box = np.array(point_size2two_point(pbox))
      z = convert_bbox_to_z(box)
      self.kf.x[:2] = z[:2]
      self.kf.x[3:5] = z[3:]
      self.history.append((box.reshape(-1,), self.kf.x[2]))
      print('predicted by kcf')
    else:
      self.kf.predict()
      #self.history.append(convert_x_to_bbox(self.kf.x))
      self.history.append(self.history[-1])
      print('predicted by kalman')

    print('[Class %s][Tracker %d]last box:'%(self.cls, self.id), self.history[max(-2, -len(self.history))][0], 'predicted box:',self.history[-1][0])
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak -= 1
    self.time_since_update += 1
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    (box, depth) = convert_x_to_bbox(self.kf.x)
    return (box, depth, self.cls, self.state)

def associate_detections_to_trackers(detections, trackers, classes, tclasses, tids, iou_threshold=0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int), np.empty((len(detections), 0), dtype=float)

  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)


  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou_elementwise(det, trk)
      if classes[d] != tclasses[t]:
        iou_matrix[d,t] = -1
      
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  print(iou_matrix)
  return matches, np.array(unmatched_detections), np.array(unmatched_trackers), iou_matrix



class Obstacles(object):
  def __init__(self,max_age=5, min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

  def associate(self, dets, depth, image):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      (pos, dp) = self.trackers[t].predict(image)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    # NMS
    detections = dets.get()
    classes = dets.get_field('classes')
    dscores = dets.get_field('scores')
    new_dets = BoxList(dets.get())
    unique_classes = list(set(classes)) 
    scores = np.zeros((dets.num_boxes(), len(set(classes)), float)
    for i in range(dets.num_boxes()):
      scores[i, unique_classes.index(classes[i])] = dscores[i]
    new_dets.add_field('classes', scores)
    keep_dboxes_idx = multi_class_non_max_suppression(new_dets, 0, NMS_THRESHOLD)
    keep_dboxes_idx = np.array(sorted(keep_dboxes_idx))
    dets.keep_indices(keep_dboxes_idx)
    detections = dets.get()
    classes = dets.get_field('classes')

    tclasses = np.array([trk.cls for trk in self.trackers])
    tids = np.array([trk.id for trk in self.trackers])
    print(detections, trks[:, :4])
    tbox_list = BoxList(trks[:, :4])
    tbox_list.add_field('classes', tclasses)
    tbox_list.add_field('ids', tids)
    overlaps = iou(dets, tbox_list)
    if np.prod(overlaps.shape) != 0:
      tscores = overlaps.max(axis=0)
      keep_tboxes_idx = non_max_suppression_slow(trks, tscores, NMS_THRESHOLD)
      keep_tboxes_idx = np.array(sorted(keep_tboxes_idx)).astype(int)
      tbox_list.keep_indices(keep_tboxes_idx)
      trks = tbox_list.get()
      tclasses = tbox_list.get_field('classes') 
      tids = tbox_list.get_field('ids') 
      overlaps = iou(dets, tbox_list)
      print(overlaps)
      matched_indices = linear_assignment(-overlaps)
    else:
      matched_indices = np.array([]).astype(int)
      if overlaps.shape[1] == 0:
        keep_tboxes_idx = []
      else:
        keep_tboxes_idx = list(range(len(trks)))
    if len(keep_tboxes_idx) != len(self.trackers):
      print('Remove %d duplicated boxes.'%(len(self.trackers) - len(keep_tboxes_idx)))
         



    matched_indices, unmatched_dets, unmatched_trks, overlaps = associate_detections_to_trackers(detections, trks, classes, tclasses, tids)

    print(tclasses)
    #update matched trackers with assigned detections
    for i, matched in enumerate(matched_indices):
      if overlaps[matched[0], matched[1]] > IOU_THRESHOLD:
        curd = detections[matched[0]].reshape(-1, )
        self.trackers[keep_tboxes_idx[matched[1]]].update(curd, classes[matched[0]], depth[curd[0]:curd[2], curd[1]:curd[3]])
        self.trackers[keep_tboxes_idx[matched[1]]].state = 'd&t'
        print('%s %d matched to detection %d'%(tclasses[matched[1]], tids[matched[1]], matched[0]) )

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      curd = detections[i].reshape(-1, )
      trk = KalmanBoxTracker(curd, classes[i], depth[curd[0]:curd[2], curd[1]:curd[3]], image) 
      trk.state = 'd'
      print('unmatched detection %d, cls %s, trackerid: %d'%(i, classes[i], trk.id) )
      self.trackers.append(trk)

    i = len(self.trackers)
    ret = BoxList()
    for trk in reversed(self.trackers):
      if((trk.time_since_update <= self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        if trk.time_since_update != 0:
          trk.state = 't'
        self.get_tracker_state(trk, ret)
        print('unmatched tracker %d, cls %s, trackerid: %d'%(i - 1, trk.cls, trk.id))
      i -= 1
        #remove dead tracklet
      if(trk.time_since_update > self.max_age):
        print('Remove tracker %d-%d, cls %s, time_since_update: %d, max_age: %d'%(trk.id, self.trackers[i].id, trk.cls, trk.time_since_update, self.max_age) )
        self.trackers.pop(i)
    return ret 

  def get_tracker_state(self, tracker, meta):
    velo = tracker.kf.x[5:]
    (box, depth, cls, state) = tracker.get_state()
    meta.add_field_data('boxes', box)
    meta.add_field_data('directions', velo)
    meta.add_field_data('depth', depth)
    meta.add_field_data('ids', tracker.id + 1)
    meta.add_field_data('classes', cls)
    meta.add_field_data('states', state)
  
  def has_obstacle(self, results, distance_threshold=0.5):
    for res in results:
      (d, meta) = res
      if res['v_y'] < 0:
         continue 
      if res['depth'] < distance_threshold:
        return False
        
      




