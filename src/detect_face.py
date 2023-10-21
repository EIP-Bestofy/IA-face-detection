import face_recognition
import cv2 as cv

class Detect_face:        
    def detect(self, frame):
        small_frame = cv.resize(frame, (0, 0), fx=0.50, fy=0.50)
        face_locations = face_recognition.face_locations(small_frame)
        return face_locations