
from yolo import YOLO
from yolo import detect_video
import sys


if __name__ == '__main__':
    video_path= sys.argv[1]
    detect_video(YOLO(), video_path)
