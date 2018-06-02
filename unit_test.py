from glue import BlindNavigator
from utility import read_video
from PIL import Image
import numpy as np
import cv2
import time

debuger = 'OFFLINE'
if debuger == 'ONLINE':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches


class Visor(object):
    def __init__(self, form = 'OFFLINE'):
        self.form = form
        self.handler = None
        self.handler2 = None
        self.rescale = True
        self.output_size = (600, 400)
        self.im_size = self.output_size


        
    def initializer(self):
        if self.form == 'OFFLINE':
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = None
            (h, w) = (None, None)
            self.handler = writer 
            
        else:
            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_axes([0,0,1,1])

            plt.ion()
            fig.show()
            fig.canvas.draw()
            self.handler = fig
            self.handler2 = ax

            
    def inliner(self, im, save_path='./test.avi'):
        self.im_size = (im.shape[1], im.shape[0])
        (w, h) =  self.output_size

        if self.rescale:
            im = cv2.resize(im, (w, h),interpolation=cv2.INTER_CUBIC)
        else:
            (h, w) = im.shape[:2]

        if self.form == 'OFFLINE' and self.handler is None:
            self.handler = cv2.VideoWriter(save_path, self.fourcc, 30, (w, h), True)
        elif self.form != 'OFFLINE':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def drawer(self, im):
        if self.form == 'OFFLINE':
            self.handler.write(im)
        else:
            self.handler2.imshow(im)
            self.handler.canvas.draw()
            self.handler2.clear()
            
    def rescale_box(self, bbox):
        if self.rescale:
            bbox = np.array([bbox[0] / self.im_size[0] * self.output_size[0], \
                             bbox[1] / self.im_size[1] * self.output_size[1], \
                             bbox[2] / self.im_size[0] * self.output_size[0], \
                             bbox[3] / self.im_size[1] * self.output_size[1]])
        bbox = bbox.astype(int)
        return bbox
    
def test_det_track():
    video_path = 'data/IMG_1051.MOV'
    save_path = 'IMG_1051_det_track.avi'

    visor = Visor(debuger)
    visor.initializer()

    navigator = BlindNavigator()
    timestamps = [time.time()]
    for data in read_video(video_path):
        timestamps.append(time.time())

        traffic_lights, detected_obstacles = navigator.detect_traffic_light(data)
        timestamps.append(time.time())
        print('Processing time: %.3f'%(timestamps[-1] - timestamps[-2]))
#         data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = visor.inliner(data, save_path)
        for (id_, bbox) in traffic_lights:
            bbox = visor.rescale_box(bbox)
            assert  id_ < len(navigator.traffic_light_pool.trackers), 'id_ not valid'
#             bbox = [bbox[0] / data.shape[1], bbox[1] / data.shape[0], bbox[2] / data.shape[1], bbox[3] / data.shape[0]]
#             print(bbox)
#             p = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, alpha=1)
#             ax.add_patch(p)
            cv2.rectangle(data, tuple(bbox[:2]),tuple(bbox[2:4]), [255, 0, 0], 3)
        visor.drawer(data)

def test_light_classifier():
    video_path = 'data/IMG_1051.MOV'
    save_path = 'IMG_1051_det_track_cls.avi'

    visor = Visor(debuger)
    visor.initializer()
    navigator = BlindNavigator()
    for data in read_video(video_path):
        traffic_lights, detected_obstacles = navigator.detect_traffic_light(data)
        light_states = navigator.color_classify_by_boxes(data, [light[1] for light in traffic_lights])
        
        data = visor.inliner(data, save_path)
        for (id_, bbox), state in zip(traffic_lights, light_states):
            
            bbox = visor.rescale_box(bbox)

            assert  id_ < len(navigator.traffic_light_pool.trackers), 'id_ not valid'
#             bbox = [bbox[0] / data.shape[1], bbox[1] / data.shape[0], bbox[2] / data.shape[1], bbox[3] / data.shape[0]]
#             print(bbox)
#             p = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, alpha=1)
#             ax.add_patch(p)
            cv2.rectangle(data, tuple(bbox[:2]),tuple(bbox[2:4]), [255, 0, 0], 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(data, '%s'%(state), (bbox[0]-1,bbox[1]-1), font, 1, color=(255,255,0), thickness=2)
        visor.drawer(data)

def test_zebra_contours():
    video_path = 'traffic light/IMG_9033.m4v'
    save_path = 'IMG_9033_zebra_track.avi'

    visor = Visor(debuger)
    visor.initializer()
    navigator = BlindNavigator()
    for data in read_video(video_path):
        p_light, detected_obstacles, traffic_lights = navigator.executor(data)
        data = visor.inliner(data, save_path)
        if p_light is not None:
            bbox = p_light.get_bbox()
            state = p_light.get_state()
            bbox = visor.rescale_box(bbox)
            # assert  id_ < len(navigator.traffic_light_pool.trackers), 'id_ not valid'
            cv2.rectangle(data, tuple(bbox[:2]),tuple(bbox[2:4]), [255, 0, 0], 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(data, '%s'%(state), (bbox[0]-1,bbox[1]-1), font, 1, color=(255,255,0), thickness=2)

        for j in navigator.zebra_contours:
            bx, by, bw, bh = cv2.boundingRect(j)
            cv2.rectangle(data, (bx, by), (bx + bw, by + bh), (180, 237, 167), -1)
        visor.drawer(data)


if __name__ == '__main__':
    test_zebra_contours()
