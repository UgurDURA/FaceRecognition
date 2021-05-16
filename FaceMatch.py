import face_recognition

image_of_Ugur=face_recognition.load_image_file('Images/Data/Ugur_DURA.jpg')
ugur_face_encoding=face_recognition.face_encodings(image_of_Ugur)[0]

image_of_team=face_recognition.load_image_file('Images/Data/Team1.jpeg')
team_face_encoding=face_recognition.face_encodings(image_of_team)[0]


results=face_recognition.compare_faces([ugur_face_encoding],team_face_encoding)

if results[0]:
    print('This is mathes')
    
else:
    print('Nope')