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



class BlindNavigator(object):
    def __init__(self):
        self.detector = YOLO('keras_yolo3/model_data/yolo.h5', anchors_path='keras_yolo3/model_data/yolo_anchors.txt', classes_path='keras_yolo3/model_data/yolo.names')
        self.depth_estimator = Depth(model_path='fcrn_depth_prediction/model/NYU_FCRN.ckpt')
#         self.zebra_detector = Zebra()
        
        self.traffic_light_pool = LightPool()
        self.STATE = ['FORWARD', 'WAIT'] 
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
        detected = self.detector.predict(image)
        detected_traffic_lights = [d for d in detected if 'traffic light' in d[0]]
        detected_obstacles = [d for d in detected if 'traffic light' not in d[0]]
        
        traffic_lights = self.traffic_light_pool.get_boxes(image, detected_traffic_lights)
        
        return traffic_lights, detected_obstacles
    
    def color_classify_by_boxes(self, image, boxes, order='012'):
        labels = []
        for box in boxes:
            cropped_image = image[box[1]:box[3], box[0]:box[2], :]
            labels.append(self.color_classify_by_patch(cropped_img, order))
        return labels
    
    def color_classify_by_patch(self, cropped_img, order='012'):
        order = [int(o) for o in order]
        img = cropped_img[:, :, order]
        return estimate_label(img.astype(np.uint8))
    
    def _is_contain_in_triangle(self, traffic_light_center, line_left, line_right):
        kl = line_left.get_k()
        bl = line_left.get_b()
        kr = line_right.get_k()
        br = line_right.get_b()
        
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
        # traffic lights are loaded from traffic_lights_pool
        # zebra_end_point: tuple (x, y), line: has k and b attribute
        is_stable, zebra_end_point, line_left, line_right, contours = self.zebra_detector.predict()
        
        if not is_stable:
            return [2] * len(traffic_lights)
        
        centers = [self._get_box_center(*traffic_light[1]) for traffic_light in traffic_lights]
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
        
        for obs in obstacles:
            xmin, ymin, xmax, ymax = obs['box']
            cropped_depth = depth[ymin: ymax, xmin: xmax]
            od = {'distance': np.median(cropped_depth.ravel())}
            obs.update(od)

    
    def _is_zebra_valid(self, obstacles, thresh=1):
        closet_obs_depth = min(obstacles, key=lambda x: x['distance'])
        if closet_obs_depth['distance'] < thresh:
            return False
        return True

    def executor(self, image):
        traffic_lights, detected_obstacles = self.detect_traffic_light(image)
        light_states = self.color_classify_by_boxes(image, [light[1] for light in traffic_lights])
        light_types = self.estimate_pedestrain_light(image, traffic_lights)
        self.traffic_light_pool.update_types(light_types)
        self.traffic_light_pool.update_states(light_states)
        plight = self.get_pedestrain_light()
    
#         if plight:
#             # find a valid green light
#             if plight.get_state() == 'G':
#                 self.action = 'FORWARD'
#                 return
    

if __name__ == '__main__':
    pass
