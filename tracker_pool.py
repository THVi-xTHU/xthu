from KCF.kcftracker import KCFTracker
from utility import *
from sklearn.utils.linear_assignment_ import linear_assignment
from bbox_transform import bbox_overlaps
from non_maximum_suppression import non_max_suppression_slow


class LightTracker(object):
    def __init__(self, image, bbox, use_hog=False, conf_threshold=0.1, state_len = 10, state_threshold = 0.9):
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
        self.state_len = state_len
        self.state_threshold = state_threshold
        self.state = 'B' 
        self.hit = 1
        self.detected = 1
        self.conf_threshold = conf_threshold
        self.confidence = 0.     # The confidence of light being a pedestrain light

    def predict(self, image):
        pbox, peak_value = self.tracker.predict(image)
        if peak_value < self.conf_threshold:
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
        if (self.pc[0] + self.pc[1]) < 10 and (self.pc[2] > self.pc[1] and self.pc[2] > self.pc[0]):
            self.type = 2
            self.confidence = 0.5
        else:
            self.confidence = self.pc[1] / (self.pc[0] + self.pc[1])      
            self.type = int(self.confidence > 0.5)
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
        self.state_hist = self.state_hist[- self.state_len:]
        print(self.state_hist)


        green = sum([ 1 for state in self.state_hist if state == 'G'])
        red = sum([ 1 for state in self.state_hist if state == 'R'])
        black = sum([ 1 for state in self.state_hist if state == 'B'])

        if green > self.state_threshold * ( red + green ):
            self.state = 'G'
        elif red > green and red > black:
            self.state = 'R'
        else:
            self.state = 'B'
        print('G: ', green, ', R: ', red, ', B: ', black, ', State_HIST: ', self.state_hist, self.state)
        
        
    def add_type(self, typ):
        self.type_hist.append(typ)
        self.pc[typ] += 1
            
    def get_hit(self):
        return self.hit

class LightPool(object):
    def __init__(self, use_hog = True, conf_threshold = 0.1, iou_threshold = 0.5, reliable_count = 5):
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
        self.conf_threshold = conf_threshold          # threshold for kcf peak value
        self.iou_threshold = iou_threshold            # threshold for det-track matching overlap iou
        self.reliable_count = reliable_count          # minimum count for a light treated as pedestrain light
        self.cur_max_times = 0                         # max times of a tracker determinated as pedestrain light 
        self.forward_max_times = 0                     # new max times
        self.output_boxes = []
        self.pedestrain_light = None
        
    def add_tracker(self, image, bbox):
        new_tracker = LightTracker(image, bbox, self.use_hog, self.conf_threshold)
        self.trackers.append(new_tracker)
        
    def count(self):
        return len(self.trackers)
    
    def remove_tracker(self):
        keep_indices = [i for i, tracker in enumerate(self.trackers) if tracker.hit >= -1 and tracker.detected >= -10 ]
        if 0 not in keep_indices:
            self.forward_max_times = 0
            self.cur_max_times = 0
            self.pedestrain_light = None
        return [ self.trackers[i] for i in keep_indices ]
        
    def predict(self, image):
        est_boxes = []
        valid_indexes = []
        for i, tracker in enumerate(self.trackers):
            bbox, pval = tracker.predict(image)
            est_boxes.append(bbox)
            valid_indexes.append(pval)
        est_boxes = np.array(est_boxes).reshape(-1, 4)
        return est_boxes, valid_indexes

    def get_boxes(self, image, dboxes, scores):
        """
        output_boxes = [(id_in_pool, (x1, y1, x2, y2)), ...]
        """
        self.remove_tracker()
        tboxes, valid_indexes = self.predict(image)
        
        keep_dboxes_idx = non_max_suppression_slow(dboxes, scores, 0.5)
        dboxes = dboxes[keep_dboxes_idx]
        
        overlaps = bbox_overlaps(tboxes, dboxes)
#         all_scores = np.ones((dboxes.shape[0] + tboxes.shape[0],))
#         all_scores[:dboxes.shape[0]] = scores[:] + 1
        
        if np.prod(overlaps.shape) != 0:
            tscores = overlaps.max(axis=1)
            keep_tboxes_idx = non_max_suppression_slow(tboxes, tscores, 0.5)
            tboxes = tboxes[keep_tboxes_idx]
            matched_indices = linear_assignment(-overlaps)
        else:
            matched_indices = np.array([])
        
        merged_boxes = []
        merged_idx = []
        d_or_t = []
        for matched in matched_indices:
            if overlaps[matched[0], matched[1]] > self.iou_threshold:
                merged_idx.append(matched[0])
                merged_boxes.append(dboxes[matched[1], :])
                d_or_t.append('d&t')
                self.trackers[matched[0]].rectify(dboxes[matched[1], :])
                self.trackers[matched[0]].detected += 1
        
        # new detected traffic lights
        for i in range(dboxes.shape[0]):
            if matched_indices.shape[0] == 0 or i not in matched_indices[:, 1]:
                merged_idx.append(self.count())
                merged_boxes.append(dboxes[i, :])
                d_or_t.append('d')
                self.add_tracker(image, dboxes[i, :])
        
        # tracked but not detected traffic lights
        for i in range(tboxes.shape[0]):
            if matched_indices.shape[0] == 0 or i not in matched_indices[:, 0]:
                if valid_indexes[i] >  self.conf_threshold:
                    merged_idx.append(i)
                    merged_boxes.append(tboxes[i, :])
                    d_or_t.append('t')
                self.trackers[i].detected -= 1
                
        print(dboxes, tboxes, matched_indices, merged_boxes, d_or_t)
        self.output_boxes = merged_boxes
        self.boxes_idx = merged_idx
        return merged_idx, merged_boxes
    
    def set_types(self, types):
        for id_, boxes, typ in zip(self.boxes_idx, self.output_boxes, types):
            self.trackers[id_].add_type(typ)
            if self.trackers[id_].pc[1] > self.forward_max_times:
                self.forward_max_times = self.trackers[id_].pc[1]
            
    def set_states(self, states):
        for id_, boxes, state in zip(self.boxes_idx, self.output_boxes, states):
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
            if self.cur_max_times < 5:
                self.pedestrain_light =  None
            else:
                self.pedestrain_light = self.trackers[0]
            return
        else:
            self.cur_max_times = self.forward_max_times
            for i, tracker in enumerate(self.trackers):
                if tracker.get_type() == 1:
                    c = tracker.pc[1]
                    if c == self.forward_max_times:
                        arg_max.append(i)
            if self.cur_max_times < 5:
                self.pedestrain_light = None
                return
            try:
                assert len(arg_max) == 1, 'ambigious! two pedestrain light in view.'
            except:
                import pdb
                pdb.set_trace()
            print('Tracker %d has been determined as pedestrain light %d times '%(arg_max[0], self.forward_max_times))
            tmp = self.trackers[arg_max[0]]
            del self.trackers[arg_max[0]]
            self.trackers.insert(0, tmp)
            self.pedestrain_light = self.trackers[0]
            return
        
    def get_pedestrain_light(self):
        return self.pedestrain_light