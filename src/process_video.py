import numpy as np
import cv2 as cv
from src.detect_landmarks import Detect_Landmarks
from src.detect_emotion import Detect_Emotion

class VideoProcessing:
    def __init__(self):
        self.landmarks_detection = Detect_Landmarks()
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
        read_status, frame = video_capture.read()
        (ih,iw) = frame.shape[:2] 
        print("Start processing the video")

        frame_count = 0  # Initialize the frame counter

        while read_status:
            frame_count += 1  # Increment the frame counter

            if frame_count % 10 == 0:  # Check if it's the 10th frame
                frame = self.process_frame(frame, type)

            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)
            cv.imshow("Bestofy", frame)
            read_status, frame = video_capture.read()

            # Stop the video processing
            key = cv.waitKey(1) & 0xFF
            if ord("q") == key:
                break
        
        self.emotion_detection.print_cluster_info()
        print("End processing the video")

    def process_frame(self, frame, type):
        if type == "face":
            states_location, locations = self.emotion_detection.detect(frame)
            self.emotion_detection.add_info_cluster(states_location, locations)
            return self.emotion_detection.display_info(states_location, locations, frame)
        elif type == "landmarks":
            landmarks = self.landmarks_detection.detect_landmarks(frame)
            self.landmarks_detection.draw_landmarks(landmarks, frame, True)

