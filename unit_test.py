import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glue import BlindNavigator
from utility import read_video
from PIL import Image
import numpy as np
import cv2
import time
import argparse
import os
import sys
from visualization import Visualizer
from hyperparams import *


class Visor(object):
    def __init__(self, form = 'OFFLINE'):
        self.form = form
        self.handler = None
        self.handler2 = None
        self.rescale = False
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
        bbox = np.asarray(bbox).astype(int)
        return bbox
    
debuger = 'OFFLINE'
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

def test_light_classifier(video_path, save_path):


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

def batch_test_zebra_contours(video_path_list):
  visor = Visor(debuger)
  visor.initializer()
  navigator = BlindNavigator()
  visualizer = Visualizer()
    
  cmap = {
        'LIGHT_WAIT': '等灯',
        'START_FORWARD':'开始前进',
        'CROSS_FORWARD':'继续前行',
        'CROSS_WAIT':'有障碍物',
        'ARRIVAL': '到达',
        
    }
  for video_path in video_path_list:
      if not video_path.endswith('.mp4'):
        continue
  
      save_path = video_path.split('.')[0] + '_processed.avi'
      if os.path.exists(save_path):
          continue
      print('Input: %s, Output:%s'%(video_path, save_path))
      for iiii, data in enumerate(read_video(video_path)):
        if data is None:
            break
        import cv2 
        print(data.shape)
        #         import pdb
        #         pdb.set_trace()
        p_light, detected_obstacles, traffic_lights = navigator.executor(data)
        data = visor.inliner(data, save_path)
    #     traffic_lights = []

        for j in navigator.zebra_contours:
    #       import pdb
    #       pdb.set_trace()
          bx, by, bw, bh = cv2.boundingRect(j)

          [bx, by, bw, bh] = visor.rescale_box([bx, by, bw, bh])
          cv2.rectangle(data, (bx, by), (bx + bw, by + bh), (220, 252, 255), -1)


        depth = navigator.depth_estimator.predict(data)

        obs_cls = detected_obstacles.get_field('classes')
        obs_boxes = detected_obstacles.get()


        data = visualizer.plot(data, obs_cls, obs_boxes, depth, navigator.is_stable, navigator.zebra_contours)

        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
        text = cmap[navigator.state]

        from PIL import Image, ImageDraw, ImageFont

        cv2_im = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
        pil_im = Image.fromarray(cv2_im)

        draw = ImageDraw.Draw(pil_im) # 括号中为需要打印的canvas，这里就是在图片上直接打印

        if 1 or p_light is not None:
    #       bbox, state = p_lights.get_bbox(), p_lights.get_state()
    #       traffic_lights.append((state, bbox))
          for bbox, state, d_or_t in zip(traffic_lights.get(), traffic_lights.get_field('states'), traffic_lights.get_field('d_or_t')):
              bbox = visor.rescale_box(bbox)
              #             assert  id_ < len(navigator.traffic_light_pool.trackers), 'id_ not valid'
              for i in range(thickness):
                draw.rectangle([tuple(bbox[:2] - i), tuple(bbox[2:4] + i)])
              label_size_ = draw.textsize('%s, %s' % (state, d_or_t), efont)
              if bbox[1] - label_size_[1] >= 0:
                    text_origin = np.array([bbox[0], bbox[1] - label_size_[1]])
              else:
                    text_origin = np.array([bbox[0], bbox[1] + 1])
              draw.rectangle([tuple(text_origin), tuple(label_size_ + text_origin)], fill=(0x00, 0x99, 0x11))
              draw.text(tuple(text_origin), '%s, %s' % (state, d_or_t), (255, 255, 255), font=efont)

        text += 'frame: %d' % iiii
        label_size = draw.textsize(text, cfont)

        draw.rectangle(
            [(0, 0), tuple(label_size)],
            fill=(0x00, 0x99, 0xFF))

        draw.text((0, 0), text, (255, 255, 255), font=cfont) # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体

        data = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        visor.drawer(data)
    
def test_zebra_contours(video_path, save_path):
  visor = Visor(debuger)
  visor.initializer()
  navigator = BlindNavigator()
  visualizer = Visualizer()
    
  cmap = {
        'LIGHT_WAIT': '等灯',
        'START_FORWARD':'开始前进',
        'CROSS_FORWARD':'继续前行',
        'CROSS_WAIT':'有障碍物',
        'ARRIVAL': '到达',
        
    }
  for iiii, data in enumerate(read_video(video_path)):
    if data is None:
        break
    import cv2 
    print(data.shape)
    #         import pdb
    #         pdb.set_trace()
    p_light, detected_obstacles, traffic_lights = navigator.executor(data)
    data = visor.inliner(data, save_path)
#     traffic_lights = []

    for j in navigator.zebra_contours:
#       import pdb
#       pdb.set_trace()
      bx, by, bw, bh = cv2.boundingRect(j)

      [bx, by, bw, bh] = visor.rescale_box([bx, by, bw, bh])
      cv2.rectangle(data, (bx, by), (bx + bw, by + bh), (220, 252, 255), -1)
    
    


    depth = navigator.depth_estimator.predict(data)
    
    obs_cls = detected_obstacles.get_field('classes')
    obs_boxes = detected_obstacles.get()
    

    data = visualizer.plot(data, obs_cls, obs_boxes, depth, navigator.is_stable, navigator.zebra_contours)
    
    data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    text = cmap[navigator.state]
    
    from PIL import Image, ImageDraw, ImageFont

    cv2_im = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)

    draw = ImageDraw.Draw(pil_im) # 括号中为需要打印的canvas，这里就是在图片上直接打印
    
    if 1 or p_light is not None:
#       bbox, state = p_lights.get_bbox(), p_lights.get_state()
#       traffic_lights.append((state, bbox))
      for bbox, state, d_or_t in zip(traffic_lights.get(), traffic_lights.get_field('states'), traffic_lights.get_field('d_or_t')):
          bbox = visor.rescale_box(bbox)
          #             assert  id_ < len(navigator.traffic_light_pool.trackers), 'id_ not valid'
          for i in range(thickness):
            draw.rectangle([tuple(bbox[:2] - i), tuple(bbox[2:4] + i)])
          label_size_ = draw.textsize('%s, %s' % (state, d_or_t), efont)
          if bbox[1] - label_size_[1] >= 0:
                text_origin = np.array([bbox[0], bbox[1] - label_size_[1]])
          else:
                text_origin = np.array([bbox[0], bbox[1] + 1])
          draw.rectangle([tuple(text_origin), tuple(label_size_ + text_origin)], fill=(0x00, 0x99, 0x11))
          draw.text(tuple(text_origin), '%s, %s' % (state, d_or_t), (255, 255, 255), font=efont)
            
    text += 'frame: %d' % iiii
    label_size = draw.textsize(text, cfont)

    draw.rectangle(
        [(0, 0), tuple(label_size)],
        fill=(0x00, 0x99, 0xFF))
    
    draw.text((0, 0), text, (255, 255, 255), font=cfont) # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体

    data = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

    visor.drawer(data)

def parse_args():
    parser = argparse.ArgumentParser(description='the Visually Impaired Assistant')
    parser.add_argument('--video_path', help='Video path of input', type=str)
    parser.add_argument('--ngpu', default='1', help='Video path of input', type=str)
    #parser.add_argument('--save_path', help='Video path to save',  type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.video_path):
        print('%s not exists')
        sys.exit(1)

#     test_zebra_contours(args.video_path, args.save_path)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    videos = []
    for c, dirs, files in os.walk(args.video_path):
        videos.extend([os.path.join(c,file) for file in files])
    batch_test_zebra_contours(videos)
