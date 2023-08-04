# import the necessary packages(TSK 572)
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
import tkinter as tk
from tkinter import ttk
import sys
import pygame


# DEFINING NECESSARY FUNCTIONS (TSK 573)

# Function to play audio from an MP3 file
mp3_file_path = "output1.mp3"


def playaudio():
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file_path)
    pygame.mixer.music.play()


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)


def popupmsg():
    tips = [
        "Tip 1: Take regular breaks and look away from the screen.",
        "Tip 2: Adjust the screen brightness and contrast for comfortable viewing.",
        "Tip 3: Use proper lighting to reduce eye strain.",
        "Tip 4: Blink frequently to keep your eyes moist.",
        "Tip 5: Maintain a proper distance between your eyes and the screen.",
        "Tip 6: Use eye drops to keep your eyes lubricated."
    ]

    # Function to show/hide tips when "Tips" button is pressed
    def show_tips():
        if tips_label.winfo_ismapped():
            tips_label.pack_forget()
        else:
            tips_label.pack(side="top", fill="x", pady=10)

    # Function to stop monitoring and close the popup window
    def stop_monitoring():
        cv2.destroyAllWindows()
        vs.stop()
        popup.destroy()
        sys.exit()

    # Function to close the popup window
    def close_window():
        popup.destroy()

    # Create the main popup window
    popup = tk.Tk()
    popup.wm_title("Monitoring...")
    popup.attributes("-topmost", True)  # Set topmost attribute to True
    style = ttk.Style(popup)
    style.theme_use('classic')
    style.configure('Test.TLabel', background='aqua')

    tips_button = ttk.Button(popup, text="Tips", command=show_tips)
    tips_button.pack(side="left", padx=10, pady=5)

    close_button = ttk.Button(
        popup, text="Continue Monitoring", command=close_window)
    close_button.pack(side="left", padx=10, pady=5)

    stop_button = ttk.Button(
        popup, text="Stop Monitoring", command=stop_monitoring)
    stop_button.pack(side="left", padx=10, pady=5)

    # Label to display tips
    tips_label = ttk.Label(popup, text="", style='Test.TLabel')
    for tip in tips:
        tips_label.configure(text=tips_label.cget("text") + "\n" + tip)

    popup.protocol("WM_DELETE_WINDOW", close_window)
    popup.mainloop()


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio

    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

 # construct the argument parse and parse the arguments


# TSK 574
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())


# TSK 575

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# COUNTER  is the total number of successive frames that have an eye aspect ratio
# less than EYE_AR_THRESH

# TOTAL  is the total number of blinks that have taken place while
# the script has been running

COUNTER = 0
TOTAL = 0

# initializing dlib's face detector (HOG-based) and then create
# the facial landmark predictor

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
print(type(predictor), predictor)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

eye_thresh = 8
before = datetime.datetime.now().minute

if not args.get("video", False):
    # Taking input from web cam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    # before =datetime.datetime.now().minute

else:
    print("[INFO] opening video file...")
    # Taking input as video file
    vs = cv2.VideoCapture(args["video"])
    time.sleep(1.0)
    # before =datetime.datetime.now().minute

# grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
            # threshold
        else:

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0

        now = datetime.datetime.now().minute
        no_of_min = now - before
        print(no_of_min, before, now)
        blinks = no_of_min * eye_thresh

        if (TOTAL < blinks - eye_thresh):
            playaudio()
            popupmsg()
            cv2.putText(frame, "Take rest for a while!!!! :D", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            TOTAL = 0
        elif (TOTAL > blinks + eye_thresh):
            playaudio()
            popupmsg()
            cv2.putText(frame, "take rest for a while!!!! :D ", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            TOTAL = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
