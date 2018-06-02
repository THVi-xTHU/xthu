# This is a glue for different components needed for this project, 
# including obstracle detection, traffic light detection, 
# single camera depth estimation, and zibra-crossing estimation. 
# Team: Tianyi Lu, Changcheng Tang, Han Shen, and Binren Tian
#

import numpy as np

from color_classify import estimate_label
from keras_yolo3.yolo import YOLO
from tracker_pool import LightPool
from fcrn_depth_prediction.depth import Depth

from Zebra import Zebra
from hyperparams import *

class BlindNavigator(object):
    def __init__(self):
        self.depth_estimator = Depth(path['FCRN'])
        self.detector = YOLO(path['YOLO'], path['YOLO_anchor'], path['YOLO_classes'])
        self.zebra_detector = Zebra()
        
        self.traffic_light_pool = LightPool()
        self.state = 'LIGHT_WAIT'
        self.alert = None
        self.zebra_contours = []
        self.is_stable = False
        """
        traffic_light_pool: list of traffic_light time stream
            traffic_light time stream: list of traffic_light, stream[i] is i-th frame's traffic light
                traffic light: (
                    id_in_pool, (xmin, ymin, xmax, ymax),

                )
                
        workflow: detect_traffic_light() -> par: estimate_pedestrain_light(), compute_distance_of_obstacles() -> make decision
        """
    
    def detect_traffic_light(self, image):
        # detected = [('cls', [array (1, 4) for box]), ...]
        detected_boxlist = self.detector.predict(image)
        detected_lights = detected_boxlist.get_specific_data('classes', 'traffic light')
        detected_obstacles = detected_boxlist.exclude_specific_data('classes', 'traffic light')
        traffic_lights = self.traffic_light_pool.get_boxes(image, detected_lights)
        print('Detected %d traffic lights, in total %d traffic lights, %d obstacles'%( \
                detected_lights.num_boxes(), traffic_lights.num_boxes(), detected_obstacles.num_boxes()))

        return traffic_lights, detected_obstacles
    
    def color_classify_by_boxes(self, image, boxes, order='012'):
        labels = []
        for box in boxes:
            box = box.astype(int)
            cropped_image = image[box[1]:box[3], box[0]:box[2], :]
            labels.append(self.color_classify_by_patch(cropped_image, order))
        return labels
    
    def color_classify_by_patch(self, cropped_img, order='012'):
        order = [int(o) for o in order]
        img = cropped_img[:, :, order]
        return estimate_label(img.astype(np.uint8))
    
    def _is_contain_in_triangle(self, traffic_light_center, line_left, line_right):
        kl = line_left[0]
        bl = line_left[1]
        kr = line_right[0]
        br = line_right[1]
        
        x, y = traffic_light_center
        
        return kl * x + bl >= y and kr * x + br >= y
    
    def _metric(self, point1, point2):
        return np.linalg.norm([p1 - p2 for p1, p2 in zip(point1, point2)], 2)
    
    def _get_valid_traffic_light(self, traffic_light_centers, intersection_point, line_left, line_right):
        is_in_tri = []
        index_of_traffic_light = -1
        for i, center in enumerate(traffic_light_centers):
            if self._is_contain_in_triangle(center, line_left, line_right) and \
            (index_of_traffic_light == -1 or 
             center[1] > traffic_light_centers[index_of_traffic_light][1]):
                index_of_traffic_light = i
                
        if index_of_traffic_light == -1:
            distances = [self._metric(center, intersection_point) for center in traffic_light_centers]
            return np.argmin(distances)
        return index_of_traffic_light
        
        
    def _get_box_center(self, xmin, ymin, xmax, ymax):
        return (xmin + xmax) / 2, (ymin + ymax) / 2
                
                
    def estimate_pedestrain_light(self, image, traffic_lights):
        # traffic lights are ndarray object
        # zebra_end_point: tuple (x, y), line: has k and b attribute
        if len(traffic_lights) == 0: return []
        _, self.is_stable, zebra_end_point, line_left, line_right, self.zebra_contours = self.zebra_detector.predict(image)
        
        if not self.is_stable:
            return [2] * len(traffic_lights)
        
        centers = [self._get_box_center(*traffic_light) for traffic_light in traffic_lights]
        ind = self._get_valid_traffic_light(centers, zebra_end_point, line_left, line_right)
        
        light_type = [0] * len(traffic_lights)
        light_type[ind] = 1
        return light_type
    

    def compute_distance_of_obstacles(self, image, obstacles):
        # obstacles: list of obstacle
        #   obstacle: {
        #     'box': (xmin, ymin, xmax, ymax),
        #     'cls': "class",
        #    }
        depth = self.depth_estimator.predict(image)

        od = []
        for obs in obstacles.get():
            xmin, ymin, xmax, ymax = obs
            cropped_depth = depth[ymin: ymax, xmin: xmax]
            od.append(np.median(cropped_depth.ravel()))
            # od.append('10')
        obstacles.add_field('distances', od)
            
    def arrive(self, image):
        if len(self.traffic_light_pool.trackers) == 0:
            return True
        return False
    
    def _has_obstacle(self, obstacles, thresh=1):
        closet_obs_depth = min(obstacles, key=lambda x: x['distance'])
        if closet_obs_depth['distance'] < thresh:
            return False
        return True

    def executor(self, image):
        traffic_lights, detected_obstacles = self.detect_traffic_light(image)
        self.compute_distance_of_obstacles(image, detected_obstacles)
        
        light_states = self.color_classify_by_boxes(image, traffic_lights.get())
        light_types = self.estimate_pedestrain_light(image, traffic_lights.get())
        traffic_lights.add_field('states', light_states)
        traffic_lights.add_field('types', light_types)

        self.traffic_light_pool.set_types(light_types)
        self.traffic_light_pool.set_states(light_states)
        self.traffic_light_pool.set_pedestrain_light()
        plight = self.traffic_light_pool.get_pedestrain_light()
        if plight:
            # find a valid green light
            if self.state == 'LIGHT_WAIT':
                if plight.get_state() == 'G':
                    self.state = 'START_FORWARD'
                    self.alert = None
            elif self.state == 'START_FORWARD':
                if plight.get_state() == 'R':
                    self.state = 'LIGHT_WAIT'
                if self._has_obstacle(detected_obstacles, obstacle_threshold):
                    self.state = 'CROSS_WAIT'
            elif self.state == 'CROSS_WAIT':
                if not self._has_obstacle(detected_obstacles, obstacle_threshold):
                    self.state = 'CROSS_FORWARD'
                if plight.get_state() == 'R':
                    self.alert = 'CROSS_RED'
            elif self.state == 'CROSS_FORWARD':
                if self._has_obstacle(detected_obstacles, obstacle_threshold):
                    self.state = 'CROSS_WAIT' 
                if plight.get_state() == 'R':
                    self.alert = 'CROSS_RED'
                if self.arrive(image):
                    self.alert = None
                    self.state = 'ARRIVE'
            elif self.state == 'ARRIVE':
                self.state = 'LIGHT_WAIT'
       
        return plight, detected_obstacles, traffic_lights
    


if __name__ == '__main__':
    pass
