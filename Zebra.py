import numpy as np
import cv2
import math
import scipy.misc
import PIL.Image
import statistics
import timeit
import glob
from sklearn import linear_model, datasets
from collections import deque

# get a line from a point and unit vectors
def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0 + scale * vx
    y1 = y0 + scale * vy
    if (x1 != x0):
        m = (y1 - y0) / (x1 - x0)
    else:
        m = (y1 - y0) * 1920
    b = y1 - m * x1
    return m, b


# the angle at the vanishing point
def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    print(len1)
    print(len2)
    a = math.acos(inner_product / (len1 * len2))
    return a * 180 / math.pi


# vanishing point - cramer's rule
def lineIntersect(m1, b1, m2, b2, W, H):
    # a1*x+b1*y=c1
    # a2*x+b2*y=c2
    # convert to cramer's system
    a_1 = -m1
    b_1 = 1
    c_1 = b1

    a_2 = -m2
    b_2 = 1
    c_2 = b2

    d = a_1 * b_2 - a_2 * b_1  # determinant
    dx = c_1 * b_2 - c_2 * b_1
    dy = a_1 * c_2 - a_2 * c_1

    if (d != 0):
        intersectionX = dx / d
        intersectionY = dy / d
    else:
        intersectionX = dx * W
        intersectionY = dy * H

    return intersectionX, intersectionY

class Zebra(object):
    def __init__(self):
        self.m_L_list = deque(maxlen=3)
        self.b_L_list = deque(maxlen=3)
        self.m_R_list = deque(maxlen=3)
        self.b_R_list = deque(maxlen=3)


    # process a frame
    def process(self, im):

        # initialize some variables
        H, W = im.shape[:2]
        x = W
        y = H

        radius = 250  # px
        # thresh = 170
        thresh = 170
        bw_width = 80

        bxLeft = []
        byLeft = []
        bxbyLeftArray = []
        bxbyRightArray = []
        bxRight = []
        byRight = []
        boundedLeft = []
        boundedRight = []

        # 1. filter the white color
        lower = np.array([thresh, thresh, thresh])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(im, lower, upper)

        # 2. erode the frame
        erodeSize = int(y / 30)
        erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize, 1))
        erode = cv2.erode(mask, erodeStructure, (-1, -1))

        # 3. find contours and  draw the green lines on the white strips
        _, contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
        # cv2.imshow("img", im)
        # cv2.waitKey(0)

        for i in contours:
            bx, by, bw, bh = cv2.boundingRect(i)

            if (bw > bw_width):
                # cv2.line(im,(bx,by),(bx+bw,by),(0,255,0),2) # draw the a contour line
                # Tianyi
                # cv2.rectangle(im, (bx, by), (bx + bw, by + bh), (180, 237, 167), -1 )  # draw the a contour line

                bxRight.append(bx + bw)  # right line
                byRight.append(by)  # right line
                bxLeft.append(bx)  # left line
                byLeft.append(by)  # left line
                bxbyLeftArray.append([bx, by])  # x,y for the left line
                bxbyRightArray.append([bx + bw, by])  # x,y for the left line
                # cv2.circle(im,(int(bx),int(by)),5,(0,250,250),2) #circles -> left line
                # cv2.circle(im,(int(bx+bw),int(by)),5,(250,250,0),2) #circles -> right line

        # cv2.imshow("img", im)
        # cv2.waitKey(0)

        # calculate median average for each line
        medianR = np.median(bxbyRightArray, axis=0)
        medianL = np.median(bxbyLeftArray, axis=0)

        bxbyLeftArray = np.asarray(bxbyLeftArray)
        bxbyRightArray = np.asarray(bxbyRightArray)

        # 4. are the points bounded within the median circle?
        for i in bxbyLeftArray:
            if (((medianL[0] - i[0]) ** 2 + (medianL[1] - i[1]) ** 2) < radius ** 2) == True:
                boundedLeft.append(i)

        boundedLeft = np.asarray(boundedLeft)

        for i in bxbyRightArray:
            if (((medianR[0] - i[0]) ** 2 + (medianR[1] - i[1]) ** 2) < radius ** 2) == True:
                boundedRight.append(i)

        boundedRight = np.asarray(boundedRight)

        # 5. RANSAC Algorithm

        # select the points enclosed within the circle (from the last part)
        bxLeft = np.asarray(boundedLeft[:, 0])
        byLeft = np.asarray(boundedLeft[:, 1])
        bxRight = np.asarray(boundedRight[:, 0])
        byRight = np.asarray(boundedRight[:, 1])

        # transpose x of the right and the left line
        bxLeftT = np.array([bxLeft]).transpose()
        bxRightT = np.array([bxRight]).transpose()

        # run ransac for LEFT
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        ransacX = model_ransac.fit(bxLeftT, byLeft)
        inlier_maskL = model_ransac.inlier_mask_  # right mask

        # run ransac for RIGHT
        ransacY = model_ransac.fit(bxRightT, byRight)
        inlier_maskR = model_ransac.inlier_mask_  # left mask

        # draw RANSAC selected circles
        # for i, element in enumerate(boundedRight[inlier_maskR]):
        # print(i,element[0])
        # cv2.circle(im,(element[0],element[1]),10,(250,250,250),2) #circles -> right line

        # for i, element in enumerate(boundedLeft[inlier_maskL]):
        # print(i,element[0])
        # cv2.circle(im,(element[0],element[1]),10,(100,100,250),2) #circles -> right line

        # 6. Calcuate the intersection point of the bounding lines
        # unit vector + a point on each line
        vx, vy, x0, y0 = cv2.fitLine(boundedLeft[inlier_maskL], cv2.DIST_L2, 0, 0.01, 0.01)
        vx_R, vy_R, x0_R, y0_R = cv2.fitLine(boundedRight[inlier_maskR], cv2.DIST_L2, 0, 0.01, 0.01)

        # get m*x+b
        m_L, b_L = lineCalc(vx, vy, x0, y0)
        m_R, b_R = lineCalc(vx_R, vy_R, x0_R, y0_R)

        # calculate intersention
        intersectionX, intersectionY = lineIntersect(m_R, b_R, m_L, b_L, W, H)

        # 7. draw the bounding lines and the intersection point
        m = radius * 10
        # if (intersectionY < H/2 ):
        #     cv2.circle(im,(int(intersectionX),int(intersectionY)),10,(0,0,255),15)
        #     cv2.line(im,(x0-m*vx, y0-m*vy), (x0+m*vx, y0+m*vy),(255,0,0),3)
        #     cv2.line(im,(x0_R-m*vx_R, y0_R-m*vy_R), (x0_R+m*vx_R, y0_R+m*vy_R),(255,0,0),3)


        # Tianyi: draw aux linesss and points
        # cv2.circle(im, (int(intersectionX), int(intersectionY)), 10, (0, 0, 255), 15)
        # cv2.line(im, (x0 - m * vx, y0 - m * vy), (x0 + m * vx, y0 + m * vy), (255, 0, 0), 3)
        # cv2.line(im, (x0_R - m * vx_R, y0_R - m * vy_R), (x0_R + m * vx_R, y0_R + m * vy_R), (255, 0, 0), 3)

        # #8. calculating the direction vector
        # POVx = W/2 #camera POV - center of the screen
        # POVy = H/2 # camera POV - center of the screen
        #
        # Dx = -int(intersectionX-POVx) #regular x,y axis coordinates
        # Dy = -int(intersectionY-POVy) #regular x,y axis coordinates
        #
        # #focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
        # focalpx = int(W * 4.26 / 6.604) #all in mm

        # end = timeit.timeit()  # STOP TIMER
        # time_ = end - start

        # print('DELTA (x,y from POV):' + str(Dx) + ',' + str(Dy))
        # return im,Dx,Dy

        return im, m_L, b_L, m_R, b_R, contours, int(intersectionX), int(intersectionY)

    def predict(self, img):
        # i = 0
        # state = ""
        is_stable = False
        contours_out = []
        # intersectionX = []
        # intersectionY = []
        # m_L = []
        # b_L = []
        # m_R = []
        # b_R = []

        try:
            # processedFrame,dx,dy = process(img)
            processedFrame, m_L, b_L, m_R, b_R, contours, intersectionX, intersectionY = self.process(img)

            # Update list
            self.m_L_list.append(m_L)
            self.b_L_list.append(b_L)
            self.m_R_list.append(m_R)
            self.b_R_list.append(b_R)

            max_m_L = np.float32(max(list(self.m_L_list)))
            min_m_L = np.float32(min(list(self.m_L_list)))
            var_m_L = max_m_L - min_m_L
            max_b_L = np.float32(max(list(self.b_L_list)))
            min_b_L = np.float32(min(list(self.b_L_list)))
            var_b_L = max_b_L - min_b_L

            max_m_R = np.float32(max(list(self.m_R_list)))
            min_m_R = np.float32(min(list(self.m_R_list)))
            var_m_R = max_m_R - min_m_R
            max_b_R = np.float32(max(list(self.b_R_list)))
            min_b_R = np.float32(min(list(self.b_R_list)))
            var_b_R = max_b_R - min_b_R

            # if (var_b_L < 100 and var_b_R < 100 and var_m_L < 0.1 and var_m_R < 0.1):
            if (var_b_L < 1000 and var_b_R < 1000 and var_m_L < 1 and var_m_R < 1):
                # if (True):
                # print('Frame %s is ok, @Stable state' % debug_count)
                is_stable = True
                for j in contours:
                    bx, by, bw, bh = cv2.boundingRect(j)
                    mid_x = bx + bw / 2
                    mid_y = by + bh / 2

                    H0, W0 = img.shape[:2]
                    W0 = W0 / 2
                    # S1
                    # sgn0 = ((m_L * W0 + b_L) > H0)
                    # sgn1 = ((m_R * W0 + b_R) > H0)
                    # sgn2 = ((m_L * mid_x + b_L) > mid_y)
                    # sgn3 = ((m_R * mid_x + b_R) > mid_y)

                    # S2
                    # sgnL = bool((m_L * mid_x + b_L) > mid_y)
                    # sgnR = bool((m_R * mid_x + b_R) > mid_y)

                    # S3
                    if (m_L > 0):
                        sgnL = bool((m_L * mid_x - mid_y + b_L) > 0)
                    else:
                        sgnL = bool((-m_L * mid_x + mid_y - b_L) > 0)
                    if (m_R > 0):
                        sgnR = bool((m_R * mid_x - mid_y + b_R) > 0)
                    else:
                        sgnR = bool((-m_R * mid_x + mid_y - b_R) > 0)

                    # cv2.circle(processedFrame, (mid_x, mid_y), 63, (0, 0, 255), -1)
                    # if ((m_L * mid_x + b_L < mid_y) and (m_R * mid_x + b_R < mid_y)):
                    if (mid_y > intersectionY):
                        # if ((sgn0 == sgn2) and (sgn1 == sgn3)):
                        if (sgnL ^ sgnR == True):
                            cv2.rectangle(processedFrame, (bx, by), (bx + bw, by + bh), (255, 255, 255), -1)  # (180, 237, 167)
                            contours_out.append(j)
                return img, is_stable, (intersectionX, intersectionY), (m_L[0], b_L[0]), (m_R[0], b_R[0]), contours_out

            else:
                # print('Frame %s is ok, @Unstable state' % debug_count)
                is_stable = False
                return img, is_stable, (0, 0), (np.float32(0), np.float32(0)), (np.float32(0), np.float32(0)), contours_out
        except Exception as e:
            # print('Failed to process frame,' + '@Unstable state')
            is_stable = False
            print('########################################################')
            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('########################################################')
            return img, is_stable, (0, 0), (np.float32(0), np.float32(0)), (np.float32(0), np.float32(0)), contours_out




if __name__ == '__main__':
    # initialization
    # cap = cv2.VideoCapture('IMG_9033.m4v')  # load a video
    # cap = cv2.VideoCapture('IMG_9043.m4v')
    cap = cv2.VideoCapture('IMG_9036.m4v')
    W = cap.get(3)  # get width
    H = cap.get(4)  # get height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('processedVideo.mp4', fourcc, 15.0, (int(W), int(H)))

    zebra_test = Zebra()

    # i = 0
    # state = ""
    # debug_count = 0
    # is_stable = False
    # contours_out = []
    # intersectionX = 0
    # intersectionY = 0
    while (cap.isOpened()):
        # debug_count += 1
        ret, frame = cap.read()
        img = scipy.misc.imresize(frame, (H, W))

        img_processed, is_stable_test, zebra_end_point_test, line_left_test, line_right_test, zebra_contours_test = zebra_test.predict(img)

        # show & save
        img_out = cv2.imshow('Processed', img_processed)
        out.write(img_out)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
