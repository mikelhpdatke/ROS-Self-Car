import cv2
import sys
import os
import numpy as np
#from matplotlib import pyplot as pp
#from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy import misc

from polydrawer import Polydrawer
from polyfitter import Polyfitter
from thresholder import Thresholder
#from undistorter import Undistorter
from warper import Warper

#undistorter = Undistorter()
thresholder = Thresholder()
warper = Warper()
polyfitter = Polyfitter()
polydrawer = Polydrawer()

def processImg(im):
    # mode = '1'
    # if mode == '0':
    #     video = 'project_video'
    #     white_output = '{}_done.mp4'.format(video)
    #     clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(0,50)
    #     white_clip = clip1.fl_image(process_image)  
    #     white_clip.write_videofile(white_output, audio=False)
    # else:
        # test_img = sys.argv[1]
        # im = misc.imread(test_img, mode='RGB')
    return process_image(im)

def process_image(frame):
    #undistorted = undistorter.undistort(frame)
    # misc.imsave('output_images/undistorted.jpg', undistorted)

    img = thresholder.threshold(frame)
    # misc.imsave('output_images/thresholded.jpg', img)

    img = warper.warp(img)
    # misc.imsave('output_images/warped.jpg', img)

    left_fit, right_fit = polyfitter.polyfit(img)
    img = polydrawer.draw(frame, left_fit, right_fit, warper.Minv)
    # misc.imsave('output_images/final.jpg', img)

    fity = np.linspace(0, 319, 320)
    left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
    pts_left = np.squeeze(pts_left, axis=0)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
    pts_right = np.squeeze(pts_right, axis=0)
    '''
    xl, yl = pts_left.T
    yl = -yl
    pp.scatter(xl, yl)
    xr, yr = pts_right.T
    yr = -yr
    pp.scatter(xr, yr)
    pp.show()
    '''
    return pts_left, pts_right

# if __name__ == '__main__':
#     main()
