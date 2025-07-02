import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import numpy as np
import argparse
import face_recognition
import cv2
import pickle
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import datetime
import sqlite3
import dlib
import dlib.cuda as cuda
dlib.DLIB_USE_CUDA = True


parser = argparse.ArgumentParser()

parser.add_argument('--path', help = 'Path of the video you want to test on.', default = 0)

args = parser.parse_args()

VIDEO_PATH = args.path


def getProfile(id):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM Person where ID="+str(id)

    cursor = conn.execute(cmd)

    profile = None

    for row in cursor:

        profile = row

    conn.close()

    return profile


def insertdata(Id, Age, Gender, Location, Time, Expression, Age_group):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM face_detection WHERE person_ID="+str(Id)

    cursor = conn.execute(cmd)

    isRecordExist = 0

    for row in cursor:

        isRecordExist = 1

    if isRecordExist == 1:

        cmd = "UPDATE face_detection SET Location="+str(Location)+" WHERE person_ID="+str(Id)

    else:

        conn.execute("INSERT INTO face_detection (person_ID, Age, Gender, Location, Time, Expression, Age_group) VALUES (?, ?, ?, ?, ?, ?, ?)", (str(Id), str(Age), str(Gender), str(Location), str(Time), str(Expression), str(Age_group)))

        conn.commit()

        print("Inserted")


def load_caffe_models():
 
    age_net = cv2.dnn.readNetFromCaffe('Age/deploy_age.prototxt', 'Age/age_net.caffemodel')
  
    return age_net


age_net = load_caffe_models()

# To run on GPU
print(dlib.DLIB_USE_CUDA)
print("Number of GPUs found: ",cuda.get_num_devices())


# Loading age and gender model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


# Loading our custom model for facial expression
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


cap = cv2.VideoCapture(VIDEO_PATH)


data = pickle.loads(open("encodings.pickle", "rb").read())

count = 0


while True:

    ret, frame = cap.read()

    orig = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


    face_ids = []

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(data["encodings"], face_encoding)

        Id = 0

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                Id = data["id"][i]
                counts[Id] = counts.get(Id, 0) + 1

            Id = max(counts, key=counts.get)

        face_ids.append(Id)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(face_locations)>0:

        faces = sorted(face_locations, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            
        (top, right, bottom, left) = faces

        roi = gray[top:bottom, left:right]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]

        emotion_probability = np.max(preds)
                        
        label = EMOTIONS[preds.argmax()]


        for (top, right, bottom, left), Id, prob in zip(face_locations, face_ids, preds):

            roii = orig[top:bottom, left:right]

            text = "{}".format(label)

            profile = getProfile(Id)


            face_img = frame[top:bottom, left:right].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        
            #Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]


            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            time = datetime.datetime.now().strftime("%I:%M:%S%p")

            cv2.putText(frame, "Id: "+str(profile[0]), (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Name: "+str(profile[1]), (left + 6, bottom + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Age: "+age, (left + 6, bottom + 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Gender: "+str(profile[3]), (left + 6, bottom + 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Age Group: "+str(profile[5]), (left + 6, bottom + 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Expression: "+text, (left + 6, bottom + 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Time: "+time, (left + 6, bottom + 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

            
            if profile[0] == 0:

                location="C:/Users/Aditya/Desktop/face/images2/0_{}.jpg".format(count)
                cv2.imwrite(location, roii)
                count += 1

            if profile[0] == 1:

                count = 1
                location="C:/Users/Aditya/Desktop/face/images2/{}.jpg".format(count)
                cv2.imwrite(location, roii)

            if profile[0] == 2:

                count = 2
                location="C:/Users/Aditya/Desktop/face/images2/{}.jpg".format(count)
                cv2.imwrite(location, roii)


            Id = profile[0]
            age_group = profile[5]
            gender = profile[3]

            insertdata(Id, age, gender, location, time, text, age_group)


        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(300, frame.shape[0]), (255,255,255), -1)

        cv2.putText(frame, "Number of persons detected: {}".format(len(face_locations)), (0,frame.shape[0] -8), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (255,0,0), 1)
        
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (320, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Recognize", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


cap.release()
cv2.destroyAllWindows()
