import numpy as np
import cv2 as cv
from src.detect_emotion import Detect_Emotion

class VideoProcessing:
    def __init__(self):
        self.emotion_detection = Detect_Emotion()


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

        while True:
            ret, frame = video_capture.read()

            frame = self.process_frame(frame, type)
            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)
            cv.imshow("Bestofy", frame)


            if not ret:
                break
            # Stop the video processing
            key = cv.waitKey(1) & 0xFF
            if ord("q") == key:
                break
        
        print("End processing the video")

    def process_frame(self, frame, type):
        states_location, locations = self.emotion_detection.detect(frame)
        return self.emotion_detection.display_info(states_location, locations, frame)


