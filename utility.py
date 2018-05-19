import numpy as np
import cv2

def two_point2point_size_array(bboxes):
    bboxes = bboxes.reshape(-1, 4)
    bboxes = np.vstack((bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]))
    return bboxes


def point_size2two_point_array(bboxes):
    bboxes = bboxes.reshape(-1, 4)
    bboxes = np.vstack((bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] + bboxes[:, 0], bboxes[:, 3] + bboxes[:, 1]))
    return bboxes


def two_point2point_size(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]



def point_size2two_point(bbox):
    return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        print('get frame %d'%i)
        yield frame
        i+=1

    cap.release()

