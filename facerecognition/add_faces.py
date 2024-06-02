import cv2
import pickle
import numpy as np
import os


# fungsi untuk open webcam add video Path: mlmodel/add_faces.py
video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('data/default.xml')

faces_data = []  # membuat data empty list

i=0

name=input('Masukan Nama: ')

# bagian ini untuk open webcam
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w :]
        resized_img=cv2.resize(crop_img,(50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)

    cv2.imshow('Frame',frame)
    vc2.waitKey(1)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break
video.release()
cv2.destroyAllWindows()


# fungsi untuk save data Path: mlmodel/add_faces.py
faces_data=np.array(faces_data)
faces_data=faces.data.reshape(100, -1)

#menambahkan data nama ke data folder kalau belum ada
if 'names.pkl' in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    names=[name]*100
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

#menambahkan data wajah ke data folder kalau belum ada
if 'faces_data.pkl' in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    names=[name]*100
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    facees=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)