import cv2
import numpy as np


class Warper:
    def __init__(self):

        src = np.float32([
            [50, 150],
            [250, 150],
            [300, 190],
            [1, 190],
        ])

        dst = np.float32([
            [80, 260],
            [160, 260],
            [160, 320],
            [80, 320],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, img):
        return cv2.warpPerspective(
            img,
            self.M,
            (img.shape[0], img.shape[1]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, img):
        return cv2.warpPersective(
            img,
            self.Minv,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )