
import cv2
from cv2 import VideoCapture
from colorsys import hsv_to_rgb
import numpy as np
from pynput.keyboard import Key, Listener

# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
inWidth  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
inHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("Camera connected.")
cv2.namedWindow("cam-original")
cv2.namedWindow("cam-test")
cv2.namedWindow("cam-diff")
cv2.namedWindow("cam-final")


display_state = 0
num_states = 2
def cycleStates():
    global display_state
    display_state += 1
    if (display_state >= num_states):
        display_state = 0
#Keyboard shit
def on_press(key):
    cycleStates()
    

def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
listener = Listener(
        on_press=on_press,
        on_release=on_release)
listener.start()

# -=-=-=-=-
def render_frame(frame, frameWidth, frameHeight, threshold):
    # output = net.forward()

# -=-=-=-

    points = []

    def fingerLine(point1, point2, color):
        cv2.line(frame, point1, point2, color, thickness=2)

    def hsv2rgb(h,s,v):
        return tuple(round(i * 255) for i in hsv_to_rgb(h,s,v))

    return frame
           

def countdown_visual(time):
    frame = 0
    while (frame < 30*time):
        s, img = cam.read()
        if s:
            cv2.imshow("cam-test", cv2.putText(img, str(frame/30), (int(inWidth/2), int(inHeight/2)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 135, 135)))
            frame += 1
        cv2.waitKey(int(1000/30))
    

background = None
countdown_visual(1)
def cap_bg():
    global background
    s, img = cam.read()
    if s:
        cv2.imshow("cam-test", img)
    background = img
cap_bg()

def imgdiff(img1, img2):
    # compute difference
    difference = cv2.absdiff(img1, img2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)# |cv2.THRESH_OTSU
    # difference[mask != 255] = [0, 0, 255]

    return difference, mask

def showOnlyDiff(img, diff, mask):
    img[mask == 255] = [255, 0, 0]
    return img

def contour(img, mask):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    return contours, hierarchy

def drawDefects(img, defects, cnt, color):
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        cv2.circle(img,far,5,color,-1)

# TODO Detect new "objects" bring those objects to the foreground.
# Essentially, you're looking to group similar, continuous differences,
# rather than just grabbing the largest differences.
# (Ex, bringing a piece of paper over a white wall)

while True:
    s, img = cam.read()
    if s:    # frame captured without any errors
        # src.upload(img)
        diff, mask = imgdiff(img, background)

        imgcpy = img.copy()
        img_final = img.copy()
        contours, hierarchy = contour(img, cv2.bitwise_not(mask))
        hull = []
        hulli = []
        hull_defects = []
        poly = []
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.drawContours(diff, contours, -1, (0, 255, 0), 2, 8, hierarchy=hierarchy)
        shape_mask = np.zeros_like(img)
        out_final = np.zeros_like(img_final)
        for i in range(min(100, len(contours))):
            hulli.append(cv2.convexHull(contours[i], returnPoints=False))
            hull.append(cv2.convexHull(contours[i], False))
            defects = None
            if (len(hull[i]) > 3):
                try:
                    # hull[i][::-1].sort(axis=0)
                    defects = cv2.convexityDefects(contours[i], hulli[i])
                except:
                    pass
            hull_defects.append(defects)

            epsilon = 0.005*cv2.arcLength(contours[i], True)
            poly.append(cv2.approxPolyDP(contours[i], epsilon, True))

        for i in range(len(hull)):
            cv2.drawContours(imgcpy, hull, i, (0, 255, 0), 3, 8)
            if (hull_defects[i] is not None):
                drawDefects(imgcpy, hull_defects[i], contours[i], (100, 255, 100))

            cv2.drawContours(imgcpy, poly, i, (0, 0, 255), 3, 8)

            if (display_state == 0):
                cv2.drawContours(shape_mask, hull, i, (255, 255, 255), -1)
            elif (display_state == 1):
                cv2.drawContours(shape_mask, poly, i, (255, 255, 255), -1)
            out_final[shape_mask == 255] = img_final[shape_mask == 255]
        cv2.imshow("cam-original", imgcpy)

        cv2.imshow("cam-test", showOnlyDiff(img, diff, mask))#render_frame(img, inWidth, inHeight, 0.05))
        cv2.imshow("cam-diff", diff)#cv2.bitwise_and(img, diff, mask)
        

        cv2.imshow("cam-final", out_final)
    cv2.waitKey(int(1000/30))

destroyWindow("cam-test")
        # imwrite("filename.jpg",img) #save image