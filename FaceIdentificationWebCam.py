import face_recognition
import os
import cv2
from numba import jit, cuda

KNOWN_FACES_DIR="Images"
TOLERANCE=0.5
MODEL="MMOD"
Frame_Thickness=1
Font_Thickness=2


camera_number = 0
video=cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)

known_faces=[]
known_names=[]

print("Loading known faces...")
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image=face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding=face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        bad_chars = ['.jpg']
        for i in bad_chars :
            filename = filename.replace(i, '')
        known_names.append(filename)

frame_width=int(video.get(3))
frame_height=int(video.get(4))
out=cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while True:
    
    ret, frame=video.read()
 
  
        
    

    locations=face_recognition.face_locations( frame, model=MODEL)
    encodings=face_recognition.face_encodings( frame, locations)
     
    for face_encoding, face_location in zip(encodings, locations):
        results=face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        name="unknown person"
       
        if True in results:
            match=known_names[results.index(True)]
            top_left=(face_location[3], face_location[0])
            bottom_right=(face_location[1], face_location[2])
            color=[0,255,0]
            cv2.rectangle(frame,top_left, bottom_right, color, Frame_Thickness)

            top_left=(face_location[3], face_location[2])
            bottom_right=(face_location[1],face_location[2]+22)
            cv2.rectangle(frame, top_left, bottom_right, color)
            cv2.putText(frame,match,(face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200), Font_Thickness)

    out.write(frame)
    cv2.imshow(name,frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        video.release()
        out.release()
        cv2.destroyAllWindows()
        break
    
            










