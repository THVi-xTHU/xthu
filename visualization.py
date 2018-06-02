import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import colorsys

from hyperparams import *
class Visualizer(object):
  def __init__(self):

    with open('keras_yolo3/model_data/yolo.names', 'r') as fr:
      all_classes = fr.read().strip().split('\n')

    hsv_tuples = [(x / len(all_classes), .4)
                  for x in range(len(all_classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    self.all_classes = all_classes
    self.colors = colors

  def depth_to_rgba(self, depth):
    _min = 0
    _max = 100
    
#     import pdb
#     pdb.set_trace()
    normed_depth = (depth - _min) / (_max - _min)

    hsv = np.ones(depth.shape[:2] + (3,))
    hsv[:, :, 0] = 0.5
    hsv[:, :, 1] = normed_depth
    hsv[:, :, 2] = 0.7
    rgba = np.zeros(depth.shape[:2] + (4,))
    rgba[:, :, :3] = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    rgba = rgba.astype(np.uint8)
    # normed pixel value
#     rgba[:, :, -1] = 0
    return rgba

  def add_mask(self, depth, boxes):
    rgba = self.depth_to_rgba(depth)
    res = np.zeros_like(rgba)
    for box in boxes:
      box = np.array(box).astype(int)
      res[box[1]: box[3], box[0]: box[2], :3] = rgba[box[1]: box[3], box[0]: box[2], :3]
      res[box[1]: box[3], box[0]: box[2], -1] = 128
    return res

  def add_zebra(self, im, contours):
    for j in contours:
      bx, by, bw, bh = cv2.boundingRect(j)
      cv2.rectangle(im, (bx, by), (bx + bw, by + bh), (180, 237, 167), -1)
    return im

  def add_boxes(self, im, clses, boxes):


    img = Image.fromarray(im)
    for cls, box in zip(clses, boxes):
      color = self.colors[self.all_classes.index(cls)]

      draw = ImageDraw.Draw(img)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
      # 第一个参数为字体文件路径，第二个为字体大小
      label_size = draw.textsize(cls, efont)

      if box[1] - label_size[1] >= 0:
        text_origin = np.array([box[0], box[1] - label_size[1]])
      else:
        text_origin = np.array([box[0], box[1] + 1])

      for i in range(thickness):
        draw.rectangle(
          [box[0] + i, box[1] + i, box[2] - i, box[3] - i],
          outline=color)
      draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=(0x00, 0x99, 0xFF))
      draw.text(text_origin, cls, fill=(255, 255, 255), font=efont)
    im = np.array(img)
    return im

  def add_traffic(self, im, lights, boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX

    colors = {'R': (255, 0, 0), 'G': (0, 255, 0), 'B': (0, 0, 0), 'Y': (255, 255, 0)}

    for cls, box in zip(lights, boxes):
      color = colors[cls]
      box = [int(b) for b in box]
      cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color, 3)
      cv2.putText(im, '%s' % cls, (box[0] - 1, box[1] - 1), font, 1, color, thickness=1)
    return im

  def plot(self, im, clses, boxes, depth, is_stable, contours):
    
    # depth = cv2.resize(depth, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)


    mask = self.add_mask(depth, boxes)

    if is_stable:
      im = self.add_zebra(im, contours)
    im = self.add_boxes(im, clses, boxes)

    # if traffic_lights:
    #     light_colors = [l[0] for l in traffic_lights]
    #     light_boxes = [l[1] for l in traffic_lights]
    #     im = self.add_traffic(im, light_colors, light_boxes)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    from PIL import Image
    im = Image.fromarray(im)
    mask = Image.fromarray(mask)
    im = Image.alpha_composite(im, mask)
    im = np.asarray(im)
    return im

# def visual_image(self,im):
# thickness = 3
# transparent_area = (50,80,100,200)
# #transparent_area = (left+thickness, top+thickness, right-thickness, bottom-thickness)
# # transparent=100  #用来调透明度，具体可以自己试
# mask = Image.new('RGBA', im.size, (0, 0, 0, 0))
# draw = ImageDraw.Draw(mask)

# draw.rectangle(transparent_area, fill=(255, 0, 0, 127))
# for i in range(thickness):
#     draw.rectangle([50+i, 80+i, 100-i, 200-i], outline=(0, 0, 255, 255))

# font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf', size=np.floor(3e-2 * im.size[1] + 2.5).astype('int32'))

# label = '{} {:.2f}'.format('Car', 0.88)
# im_draw = ImageDraw.Draw(im)
# label_size = im_draw.textsize(label, font)

# # top, left, bottom, right = transparent_area
# # top = max(0, np.floor(top + 0.5).astype('int32'))
# # left = max(0, np.floor(left + 0.5).astype('int32'))
# # bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
# # right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
# # print(label, (left, top), (right, bottom))
# if top - label_size[1] >= 0:
#     text_origin = np.array([left, top - label_size[1]])
# else:
#     text_origin = np.array([left, top + 1])
# im_draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=(0, 0, 255, 255))
# im_draw.text(text_origin, label, fill=(255, 255, 255), font=font)
# # del draw




# im = Image.alpha_composite(im, mask)
# im = im.convert("RGB")  # Remove alpha for saving in jpg format.
# im.save('test_processed.png')
# im.show()



















