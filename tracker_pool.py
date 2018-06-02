from KCF.kcftracker import KCFTracker
from utility import *
from sklearn.utils.linear_assignment_ import linear_assignment
from non_maximum_suppression import non_max_suppression_slow

from util.box_list import BoxList
from util.box_list_ops import *
from hyperparams import *

class LightTracker(object):
    def __init__(self, image, bbox, use_hog=False):
        """ type: 0: not pedestrain light, 1: pedestrain light, 2: not determined
            state: 'G', 'R', 'B'
        """
        self.tracker = KCFTracker(use_hog, True, True) # hog, fixed_window, multiscale
        self.bbox = two_point2point_size(bbox)
        self.tracker.init(self.bbox, image)
        self.type_hist = []
        self.state_hist = []
        self.pc = [0, 0, 0] # number of pedestrain, non-pedestrain, not-determined
        self.type = 0 
        self.state = 'B'
        self.verified = False
        self._hit = 1
        self._chit = 1
        self._detected = 1
        self.confidence = 0.     # The confidence of light being a pedestrain light
        
    @property
    def hit(self):
        return self._hit
    
    @hit.setter
    def hit(self, val):
        if self._hit < TYPE_LEN:
            self._hit = val

             
    @property
    def chit(self):
        return self._chit
    
    @chit.setter
    def chit(self, val):
        if self._chit < TYPE_LEN:
            self._chit = val
        if self._chit >= MIN_DET_COUNT:
            self.verified = True
            
    @property
    def detected(self):
        return self._detected
    
    @detected.setter
    def detected(self, val):
        if self._detected < TYPE_LEN:
            self._detected = val

    def predict(self, image):
        pbox, peak_value = self.tracker.predict(image)
        if peak_value < TCONF_THRESHOLD:
            self.hit -= 1
        else:
            self.bbox = pbox
        return point_size2two_point(pbox), peak_value

    def rectify(self, bbox):
        self.bbox = two_point2point_size(bbox)
        self.hit += 1
     
    def check_pedestrain(self):
        length = len(self.type_hist)
        if length == 0:
            return 0
        if len(self.type_hist) < TYPE_WAIT_LEN:
            self.type = 2
            self.confidence = CONF_AVERAGE
        else:
            self.confidence = self.pc[1] / (self.pc[0] + self.pc[1])
            if self.confidence > TYPE_THRESHOLD:
                self.type = 1
            elif self.confidence < 1 - TYPE_THRESHOLD:
                self.type = 0
            else:
                self.type = 2
        return   

    def get_confidence(self):
        self.check_pedestrain()
        return self.confidence
    

    def get_type(self):
        self.check_pedestrain()
        return self.type

    def get_state(self):
        return self.state
    
    def get_bbox(self):
        return point_size2two_point(self.bbox)
    
    def add_state(self, state):
        self.state_hist.append(state)
        print(self.state_hist)
        self.state_hist = self.state_hist[- STATE_LEN:]
        print(self.state_hist)


        green = sum([ 1 for state in self.state_hist if state == 'G'])
        red = sum([ 1 for state in self.state_hist if state == 'R'])
        black = sum([ 1 for state in self.state_hist if state == 'B'])

        if green > STATE_THRESHOLD * ( red + green ):
            self.state = 'G'
        elif red > green and red > black:
            self.state = 'R'
        else:
            self.state = 'B'
        print('G: ', green, ', R: ', red, ', B: ', black, ', State_HIST: ', self.state_hist, self.state)
        
        
    def add_type(self, typ):
        if typ != 2:
            self.type_hist.append(typ)
            self.pc[typ] += 1
        if len(self.type_hist) == TYPE_LEN:
            pop_typ = self.type_hist.pop(0)
            self.pc[pop_typ] -= 1
            

    def get_hit(self):
        return self.hit

class LightPool(object):
    def __init__(self, use_hog = True):
        """
        Workflow: get_boxes() -> set_types() and set_states() -> set_pedestrain_light()
        The light pool: 
            The light pool is maintained since the first traffic light detected. When a new frame comes, 
            the get_boxes() function is called to get all possible traffic lights (det + track). The new detections
            are added into TrackPool, and trackers are updated by matched detection boxes. 
            When color classification and pedestrain light classification is finished by exteral decision maker, it calls 
            set_types() and set_states() to update trackers' states. 
            Afterwards, set_pedestrain_light() is called to determine whether there's a pedestrain light, 
            if yes put the tracker in the self.trackers queue front, and return the tracker, 
            else do no shuffling and return None. 
        """
        self.trackers = []
        self.use_hog = use_hog
        self.cur_max_times = 0                         # max times of a tracker determinated as pedestrain light 
        self.forward_max_times = 0                     # new max times
        self.output_boxlist = None
        self.pedestrain_light = None
        
    def add_tracker(self, image, bbox):
        new_tracker = LightTracker(image, bbox)
        self.trackers.append(new_tracker)
        
    def count(self):
        return len(self.trackers)
    
    def remove_tracker(self):
        keep_indices = [i for i, tracker in enumerate(self.trackers) if tracker.hit >= -1 and tracker.detected >= -10 ]
        if self.forward_max_times > TYPE_WAIT_LEN * TYPE_THRESHOLD and 0 not in keep_indices and len(self.trackers) > 0:        
            print('Remove current tracker, the tracker has hit %d, detected %d'%(
                self.trackers[0].hit, self.trackers[0].detected))
            print('Tracker Info: forward pc %d, cur pc %d, tracker pc %d:'%(self.forward_max_times, self.cur_max_times, self.trackers[0].pc[1]), self.trackers[0].get_bbox())
        self.trackers = [self.trackers[i] for i in keep_indices]
        if 0 not in keep_indices:
            # TODO, refind the forward max time
            if keep_indices:
                cur_tracker = np.array([tracker.pc[1] for tracker in self.trackers])
                max_id = cur_tracker.argmax()
                self.trackers[0], self.trackers[max_id] = self.trackers[max_id], self.trackers[0]
                self.forward_max_times = cur_tracker[max_id]
                self.cur_max_times = cur_tracker[max_id]
            else:
                self.forward_max_times = 0
                self.cur_max_times = 0
        return 
        
    def predict(self, image):
        est_boxes = []
        valid_indices = []
        for i, tracker in enumerate(self.trackers):
            bbox, pval = tracker.predict(image)
            est_boxes.append(bbox)
            valid_indices.append(pval)
        est_boxes = np.array(est_boxes).reshape(-1, 4)
        tbox_list = BoxList(est_boxes)
        tbox_list.add_field('valid_indices', valid_indices)
        return tbox_list

    def get_boxes(self, image, dbox_list):
        """
        output_boxes = [(id_in_pool, (x1, y1, x2, y2)), ...]
        """
        self.remove_tracker()
        tbox_list = self.predict(image)

        dboxes = dbox_list.get()
        dscores = dbox_list.get_field('scores')
        
        keep_dboxes_idx = non_max_suppression_slow(dboxes, dscores, NMS_THRESHOLD)

        dbox_list.keep_indices(sorted(keep_dboxes_idx))

        overlaps = iou(tbox_list, dbox_list)
        print(dboxes, '\n', tbox_list.get(), overlaps)

        if np.prod(overlaps.shape) != 0:
            tscores = overlaps.max(axis=1)

            tbox_list.add_field('scores', tscores)
            keep_tboxes_idx = non_max_suppression_slow(tbox_list.get(), tscores, NMS_THRESHOLD)

            tbox_list.keep_indices(sorted(keep_tboxes_idx))
            overlaps = iou(tbox_list, dbox_list)
            matched_indices = linear_assignment(-overlaps)
        else:
            matched_indices = np.array([])
        
        merged_boxes = []
        merged_idx = []
        d_or_t = []
        keep_indices = []
        for i, matched in enumerate(matched_indices):
            if overlaps[matched[0], matched[1]] > IOU_THRESHOLD:
                merged_idx.append(matched[0])
                merged_boxes.append(dboxes[matched[1], :])
                d_or_t.append('d&t')
                self.trackers[matched[0]].rectify(dboxes[matched[1], :])
                self.trackers[matched[0]].detected += 1
                self.trackers[matched[0]].chit += 1
                keep_indices.append(i)
        
        matched_indices = matched_indices[keep_indices]
        print(overlaps)
        print(matched_indices)
        
        # new detected traffic lights
        for i in range(dbox_list.num_boxes()):
            if matched_indices.shape[0] == 0 or i not in matched_indices[:, 1]:
                merged_idx.append(self.count())
                merged_boxes.append(dboxes[i, :])
                d_or_t.append('d')
                self.add_tracker(image, dboxes[i, :])


        tboxes = tbox_list.get()
        valid_indices = tbox_list.get_field('valid_indices')
        # tracked but not detected traffic lights
        for i in range(tbox_list.num_boxes()):
            if matched_indices.shape[0] == 0 or i not in matched_indices[:, 0]:
                if self.trackers[i].verified and valid_indices[i] >  TCONF_THRESHOLD:
                    merged_idx.append(i)
                    merged_boxes.append(tboxes[i, :])
                    d_or_t.append('t')
                self.trackers[i].detected -= 1
                self.trackers[i].chit = 0

                
        print(dboxes, tboxes, matched_indices, merged_boxes, d_or_t)
        self.output_boxlist = BoxList({
          'boxes': merged_boxes,
          'index': merged_idx,
          'd_or_t': d_or_t
        })
        return self.output_boxlist
    
    def set_types(self, types):
        for id_, typ in zip(self.output_boxlist.get_field('index'), types):
            self.trackers[id_].add_type(typ)
            if self.trackers[id_].pc[1] > self.forward_max_times:
                self.forward_max_times = self.trackers[id_].pc[1]
            
    def set_states(self, states):
        for id_, state in zip(self.output_boxlist.get_field('index'),  states):
            self.trackers[id_].add_state(state)

           
    def set_pedestrain_light(self):
        """
        Return pedestrain light: if not sure, return None, else return the tracker. 
        It shuffles the tracker order to put the pedestrain light in the queue front.
        """
        print('Cur Max Pedestrain Light vote %d, Forward Max Pedestrain Light vote %d'%(self.cur_max_times, self.forward_max_times))
        arg_max = []
        if len(self.trackers) == 0:
            self.pedestrain_light = None
            return
        if self.forward_max_times == self.cur_max_times:
            # if initialize or when no tracker with larger confidence if found
            if self.cur_max_times >= TYPE_WAIT_LEN * TYPE_THRESHOLD and  self.trackers[0].type == 1:
                self.pedestrain_light =  self.trackers[0]
            else:
                self.pedestrain_light =  None

            return
        else:
            # a larger pedestrain number found: 
            self.cur_max_times = self.forward_max_times
            for i, tracker in enumerate(self.trackers):
                if tracker.get_type() == 1:
                    c = tracker.pc[1]
                    if c == self.forward_max_times:
                        arg_max.append(i)
            if not arg_max:
                return
            
            if len(arg_max) > 1:
                print('[LOG] find %d feasible light, cannot decide which pedestrain light is better, wait...'%len(arg_max))
                return
            print('Tracker %d has been determined as pedestrain light %d times '%(arg_max[0], self.forward_max_times))
            # insert the selected tracker to front of queue
            tmp = self.trackers[arg_max[0]]
            del self.trackers[arg_max[0]]
            self.trackers.insert(0, tmp)
            self.pedestrain_light = self.trackers[0]
            return
        
    def get_pedestrain_light(self):
        return self.pedestrain_light