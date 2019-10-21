import cv2
from subprocess import check_output
import os
import numpy as np

# Load all the images
# print(check_output(["ls", "C://pythoncode//HandGesture-master//HandGesture-master//temp_images//"]).decode("utf8"))
folder = "temp_images//"
only_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
# Capturing video from Camera
cap = cv2.VideoCapture(0)


# --------------------- Function definition for Frame and Matching of Gestures --------------------- #


def gesture(temp_file, gesture_name, gesture_title, percision):
    # Reading image for Victory-template
    print(temp_file)
    vic_temp1 = cv2.imread("temp_images//" + temp_file, 0)
    vic_temp2 = cv2.imread("temp_images//" + temp_file, 1)
    w, h = vic_temp1.shape[::-1]
    # Change color of image to YCR
    ycc1 = cv2.cvtColor(vic_temp2,cv2.COLOR_BGR2YCR_CB)
    vic_bw_temp, cr1, cb1 = cv2.split(ycc1)
    # Change color of image to binary image
    cv2.threshold(cr1, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, vic_bw_temp)
    kernel1 = np.ones((9,9), np.uint8)
    # Binary Image
    vic_bw_temp = cv2.morphologyEx(vic_bw_temp,cv2.MORPH_CLOSE,kernel1)
    # cv2.imshow("gesture_template", vic_bw_temp)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        cv2.flip(frame,1,frame)
        # Change color of image to YCR
        ycc = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
        # Change color of image to bw
        vic_bw_frame, cr, cb = cv2.split(ycc)
        # Change color of image to binary
        cv2.threshold(cr, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU, vic_bw_frame)
        kernel = np.ones((9,9), np.uint8)
        vic_bw_frame = cv2.morphologyEx(vic_bw_frame,cv2.MORPH_CLOSE,kernel)
        # Apply template Matching
        res = cv2.matchTemplate(vic_bw_frame, vic_bw_temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Print Precision value for Victory
        print(max_val)
        # if max_val is great than 0.65 then show a rec
        if max_val > percision:
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # Rectangle around the detected gesture
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
            # Text for detected gesture
            cv2.putText(frame,gesture_name,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.rectangle(vic_bw_frame, top_left, bottom_right, (0, 0, 255), 4)
            cv2.putText(vic_bw_frame, gesture_name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Display the resulting frame
        cv2.imshow(gesture_title,frame)
        # Show the Binary version of the template image and camera video
        # cv2.imshow(gesture_title, vic_bw_frame)

        # Run the frame until user press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --------------------- Function calling for Gesture Recognition --------------------- #

gesture(only_files[0], 'Respect', 'Respect-Gesture', 0.67)
gesture(only_files[1], 'Like', 'Like-Gesture', 0.80)
gesture(only_files[2], 'Victory', 'Victory-Gesture', 0.65)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
