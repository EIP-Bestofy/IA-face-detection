import numpy as np
import cv2 as cv
from src.detect_landmarks import LandmarksDetector
from src.detect_emotion import detect_emotion
from src.display_info import display_info

class VideoProcessing:
    def __init__(self):
        self.landmarks_detector = LandmarksDetector()

    def process_video(self, video_path, type):
        video_capture = cv.VideoCapture(video_path)
        if not video_capture.isOpened():
            print("Fail to open the video.\n\tPath to video entered: ", video_path)
            return
        print("Succes to open the video: ", video_path)

        self.process_capture(video_capture, type)
        video_capture.release()
        cv.destroyAllWindows()

    def process_capture(self, video_capture, type):
        read_status, frame = video_capture.read()
        (ih,iw) = frame.shape[:2] 
        print("Start processing the video")

        while read_status:

            frame = self.process_frame(frame, type)
            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)
            cv.imshow("Bestofy", frame)
            read_status, frame = video_capture.read()

            # Stop the video processing
            key = cv.waitKey(1) & 0xFF
            if ord("q") == key:
                break

        print("End processing the video")

    def process_frame(self, frame, type):
        if type == "face":
            states_location, locations = detect_emotion(frame)
            return display_info(states_location, locations, frame)
        elif type == "landmarks":
            landmarks = self.landmarks_detector.detect_landmarks(frame)
            self.landmarks_detector.draw_landmarks(landmarks, frame, True)

