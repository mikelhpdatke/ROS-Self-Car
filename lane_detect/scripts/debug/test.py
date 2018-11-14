import cv2
import numpy as np
from scipy.stats import itemfreq


def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]



def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (5,5), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3,3), 0) 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,128,0])
    upper_blue = np.array([215,255,255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0,0,128], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only blue colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0,0,0], dtype=np.uint8)
    upper_black = np.array([170,150,50], dtype=np.uint8)

    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    return mask

def runn(success, frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(gray,1)
    # img = chay(frame)
    image=frame.copy()
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image,300)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))
    img= binary_image.copy()

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=100, param2=40)
    cv2.imshow('imageokok.jpg',img)                          
    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]

            dominant_color = get_dominant_color(square, 2)
            if dominant_color[2] > 100:
                print("STOP")
            elif dominant_color[0] > 80:
                zone_0 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                # cv2.imshow('Zone0', zone_0)
                zone_0_color = get_dominant_color(zone_0, 1)

                zone_1 = square[square.shape[0]*1//8:square.shape[0]
                                * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                # cv2.imshow('Zone1', zone_1)
                zone_1_color = get_dominant_color(zone_1, 1)

                zone_2 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                # cv2.imshow('Zone2', zone_2)
                zone_2_color = get_dominant_color(zone_2, 1)

                if zone_1_color[2] < 60:
                    if sum(zone_0_color) > sum(zone_2_color):
                        print("LEFT")
                    else:
                        print("RIGHT")
                else:
                    if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                        print("FORWARD")
                    elif sum(zone_0_color) > sum(zone_2_color):
                        print("FORWARD AND LEFT")
                    else:
                        print("FORWARD AND RIGHT")
            else:
                print("N/A")

        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('camera', frame)

def chay(image):
    # image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    # cv2.imshow("camera", edged)
    # cv2.waitKey(0)
    return edged

if __name__ == '__main__':

    cameraCapture = cv2.VideoCapture(0) 
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', onMouse)

    # Read and process frames in loop
    # success, frame = cameraCapture.read()

    success= True;
    frame= cv2.imread('anh3.jpg')
    # chay(frame)
    clicked = False

    while success and not clicked:
        runn(success, frame)
        cv2.waitKey(1)

        # success, frame = cameraCapture.read()

        success= True;
        frame= cv2.imread('anh3.jpg')
    
    cv2.destroyAllWindows()
    cameraCapture.release