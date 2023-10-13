import face_recognition
import cv2 as cv

def detect_face(frame):
    small_frame = cv.resize(frame, (0, 0), fx=0.50, fy=0.50)
    face_locations = face_recognition.face_locations(small_frame)
    return face_locations