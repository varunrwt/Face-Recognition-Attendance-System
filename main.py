import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

#load camera
cap = cv2.VideoCapture(0)

while True:
   ret,frame = cap.read()
   face_location,face_names = sfr.detect_known_faces(frame)
   for face_location,face_names in zip(face_location,face_names):
        y1, x2, y2, x1 = face_location[0], face_location[1], face_location[2], face_location[3]
        cv2.putText(frame, face_names,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
   cv2.imshow("frame",frame)
   key = cv2.waitKey(1)
   if key==27:
      break

cap.release()
cap.destroyAllWindows()
