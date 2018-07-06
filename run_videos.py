import os
from unit_test import test_zebra_contours

if __name__ == '__main__':
    dir = 'Trim/'
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not file.endswith('.mp4'):
                continue
            video_path = os.path.join(root,file)
            save_path = file[:-4] + '_processed.mp4'
            if not os.path.exists(video_path) or os.path.exists(save_path):
                continue
            print('Input: %s, Output:%s'%(video_path, save_path))
            try:
                test_zebra_contours(video_path, save_path)
            except Exception as e:
                print(e)
                pass
