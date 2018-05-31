import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
im = Image.open("test.jpg")
im = im.convert("RGBA")


#def visual_image(self,im):
thickness = 3
transparent_area = (50,80,100,200)
#transparent_area = (left+thickness, top+thickness, right-thickness, bottom-thickness)
# transparent=100  #用来调透明度，具体可以自己试
mask = Image.new('RGBA', im.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(mask)

draw.rectangle(transparent_area, fill=(255, 0, 0, 127))
for i in range(thickness):
    draw.rectangle([50+i, 80+i, 100-i, 200-i], outline=(0, 0, 255, 255))

font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * im.size[1] + 2.5).astype('int32'))

label = '{} {:.2f}'.format('Car', 0.88)
im_draw = ImageDraw.Draw(im)
label_size = im_draw.textsize(label, font)

top, left, bottom, right = transparent_area
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
print(label, (left, top), (right, bottom))
if top - label_size[1] >= 0:
    text_origin = np.array([left, top - label_size[1]])
else:
    text_origin = np.array([left, top + 1])
im_draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=(0, 0, 255, 255))
im_draw.text(text_origin, label, fill=(255, 255, 255), font=font)
# del draw


# Draw Zebra Lines
# if is_stable == True:
#     for j in contours_out:
#         cv2.rectangle(im, (bx, by), (bx + bw, by + bh), (180, 237, 167), -1)  # draw the a contour line

# Alpha composite the two images together.
im = Image.alpha_composite(im, mask)
im = im.convert("RGB")  # Remove alpha for saving in jpg format.
im.save('test_processed.png')
im.show()



















