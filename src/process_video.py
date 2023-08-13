import numpy as np
import cv2 as cv
from src.detect_face import FaceDetector
from src.detect_landmarks import LandmarksDetector

class VideoProcessing:
    def __init__(self):
        self.landmarks_detector = LandmarksDetector()
        self.face_detector = FaceDetector()

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
        print("Start processing the video")

        while read_status:

            frame = self.process_frame(frame, type)

            cv.imshow("Bestofy", frame)
            read_status, frame = video_capture.read()

            # Stop the video processing
            key = cv.waitKey(1) & 0xFF
            if ord("q") == key:
                break

        print("End processing the video")

    def process_frame(self, frame, type):
        if type == "face":
            small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            face_locations = self.face_detector.detect_face(small_frame)
            frame = self.face_detector.draw_square_at_locations(face_locations, frame)
        elif type == "landmarks":
            landmarks = self.landmarks_detector.detect_landmarks(frame)
            frame = self.landmarks_detector.draw_landmarks(landmarks, frame, True)

        return frame
