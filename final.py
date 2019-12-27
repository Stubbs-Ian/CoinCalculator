#Special thanks to PyImageSearch for teaching me about
#   image detection and also to
#Tiziano Fiorenzani for camera calibration
	
@book{rosebrock_rpi4cv,
  author={Rosebrock, Adrian and Hoffman, Dave and McDuffee, David,
	and Thanki, Abhishek and Paul, Sayak},
  title={Raspberry Pi for Computer Vision},
  year={2019},
  edition={1.0.0},
  publisher={PyImageSearch.com}
}

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#---------------------------------------
# If circumference is within range the return
# coin value
#---------------------------------------
def amount(w):
    amount = 0.00
    if (w < 0.700):
        amount = 0
    elif (w < 0.750):
        amount = 0.10
    elif (w < 0.835):
        amount = 0.01
    elif (w < 0.955):
        amount = 0.05
    elif (w < 1.043):
        amount = 0.25
    return amount

#---------------------------------------
# Finding the midpoint of edge
#---------------------------------------
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#---------------------------------------
# Biggest part, will calibrate/detect edges/ 
# and a bunch more good stuff
#---------------------------------------
def total():
    total = 0.00 #will be returned to keep track of running total
    
    #load in camera matrix and distortion coefficient from initial calibration
    fname1 = 'cameraMatrix.txt'
    fname2 = 'cameraDistortion.txt'
    mtx = np.loadtxt(fname1, delimiter=',')
    dist1 = np.loadtxt(fname2, delimiter=',')
 
    #Read in image that was captured
    image = cv2.imread("/home/pi/CS499/image0.png")
    h, w = image.shape[:2]
    
    #set up to correct distortion of image that was captured. 
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist1, (w,h),1,(w,h))
    #undistort image and then write undistorted image over the existing file. 
    dst = cv2.undistort(image, mtx, dist1, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('image1.png',dst)
    image = cv2.imread("/home/pi/CS499/image1.png")
    
    #grey and blur image to start process of edge detection
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (7, 7), 0)
    
    #detect edged after image has been blurred and greyed. 
    edged = cv2.Canny(grey, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    #create a list of contoured edges. 
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    
    pixelsPerMetric = None #this is needed to help determin left most image
    
    #show the user what edged have been detected. 
    cv2.imshow("Image", edged)
    cv2.waitKey(0)
    
    #cycle through every edge that was detected
    for c in cnts:
        #if the contour area is small then it is likely noise and will be skipped
        if cv2.contourArea(c) < 100:
            continue
        orig = image.copy()
        box = cv2.boxPoints(cv2.minAreaRect(c))
        
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        #Find the middle of the detected circle
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        #Draw a box around the circle that was detected by edge detection. 
        #This is just for visualization and is not exactly needed. 
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        #create a horizontal and vertical line through the coin
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                                                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                                                 (255, 0, 255), 2)

        #to calculate the distance between the two edges on the horizonal plane
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        #Setting baseline size for left most detected image
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 0.955
        
        dimB = dB / pixelsPerMetric
        
        #setting detected size of image on screen to see
        cv2.putText(orig, "{:.2f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        
        total += amount(dimB)
        
        cv2.imshow("Image", orig)
        cv2.waitKey(0)
        
    return total

#---------------------------------------
# Main function and will run most of the other
# functions in the program. 
#---------------------------------------
def main():

    # Letitng user know that camera is starting as a terminal message
    print("[INFO] starting video stream...")

    # Setting up camera settings and starting camera
    vs = VideoStream(usePiCamera=True, resolution=(640, 480)).start()
    time.sleep(2.0)

    circle_count = 0.00 #this will be used to keep a running total
    
    # loop over the frames from the video stream
    while True:
        # grab the frame from the  video stream and resize it to have a
        # maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400, inter=cv2.INTER_NEAREST)

        # Show video frame and wait for input key
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If input key is a - take a picture and run edge detection
        if key == ord("a"):
            
            img_count = 0 #TODO: incriment number to save all images. 
            
            #Save and image to current working directory. 
            # this image will be used for edge detection. 
            img_name = "image{}.png".format(img_count)
            cv2.imwrite(img_name, frame)
            
            circle_count += total()
            print("[INFO] current total is: ${:.2f}".format(circle_count))
     
        # if the `q` key was pressed, break from the loop and close program
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

    #print the final total that was tallied
    print("[INFO] final total is: ${:.2f}".format(circle_count))

#---------------------------------------
# Start of the program
#---------------------------------------    
if __name__ == "__main__":
    main()