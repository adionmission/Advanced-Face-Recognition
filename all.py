import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from collections import Counter
import argparse
import face_recognition
import cv2
import csv
import pandas as pd
import math
import glob
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import datetime
import sqlite3
import matplotlib.pyplot as plt
import dlib
import dlib.cuda as cuda
dlib.DLIB_USE_CUDA = True


parser = argparse.ArgumentParser()

parser.add_argument('--video_path', help = 'Path of the video you want to test on.', default = 0)

parser.add_argument('--image_dir', help="Path to image directory only", default=None)

args = parser.parse_args()

VIDEO_PATH = args.video_path

image_dir = args.image_dir


def getProfile(id):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM Person where ID="+str(id)

    cursor = conn.execute(cmd)

    profile = None

    for row in cursor:

        profile = row

    conn.close()

    return profile


def insertdata(Id, Age, Gender, Location, Time, Expression, Age_group, Confidence, Count):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM face_detection WHERE person_ID="+str(Id)

    cursor = conn.execute(cmd)

    isRecordExist = 0

    for row in cursor:

        isRecordExist = 1

    if isRecordExist == 1:

        cmd = "UPDATE face_detection SET Location="+str(Location)+" WHERE person_ID="+str(Id)

    else:

        conn.execute("INSERT INTO face_detection (person_ID, Age, Gender, Location, Time, Expression, Age_group, Confidence, Count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (str(Id), str(Age), str(Gender), str(Location), str(Time), str(Expression), str(Age_group), str(Confidence), str(Count)))

        conn.commit()

        print("Inserted")


def load_caffe_models():
 
    age_net = cv2.dnn.readNetFromCaffe('Age/deploy_age.prototxt', 'Age/age_net.caffemodel')
  
    return age_net


age_net = load_caffe_models()


@contextmanager
def video_capture(*args, **kwargs):
    
    cap = cv2.VideoCapture(*args, **kwargs)
    
    try:
        
        yield cap
    
    finally:
        
        cap.release()


def yield_videos():

    with video_capture(VIDEO_PATH) as cap:
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:

            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            
            h, w, _ = img.shape
            
            r = 960 / max(w, h)
            
            yield cv2.resize(img, (int(w * r), int(h * r)))


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    
    if face_distance.all() > face_match_threshold:
        
        range = (1.0 - face_match_threshold)
        
        linear_val = (1.0 - face_distance) / (range * 2.0)
        
        return linear_val
    
    else:
        
        range = face_match_threshold
        
        linear_val = 1.0 - (face_distance / (range * 2.0))
        
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


# To run on GPU
print(dlib.DLIB_USE_CUDA)
print("Number of GPUs found: ",cuda.get_num_devices())


# Loading age model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


# Loading our custom model for facial expression
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


data = pickle.loads(open("encodings.pickle", "rb").read())


image_generator = yield_images_from_dir(image_dir) if image_dir else yield_videos()

count = 0
countt = 0
time = 5000  # 5000 specifies 5 seconds

for img in image_generator:

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig = img.copy()

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


    face_ids = []

    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(data["encodings"], face_encoding)

        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)

        Id = 0

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                Id = data["id"][i]
                counts[Id] = counts.get(Id, 0) + 1

            Id = max(counts, key=counts.get)

        face_ids.append(Id)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(face_locations)>0:

        faces = sorted(face_locations, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                
        (top, right, bottom, left) = faces

        roi = gray[top:bottom, left:right]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
                            
        label = EMOTIONS[preds.argmax()]


        for (top, right, bottom, left), Id, prob in zip(face_locations, face_ids, preds):

            roii = orig[top:bottom, left:right]

            text = "{}".format(label)

            profile = getProfile(Id)


            face_img = img[top:bottom, left:right].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            confidence = face_distance_to_conf(face_distances)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            Time = datetime.datetime.now().strftime("%I:%M:%S%p")

            cv2.putText(img, "Id: "+str(profile[0]), (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Name: "+str(profile[1]), (left + 6, bottom + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Age: "+age, (left + 6, bottom + 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Gender: "+str(profile[3]), (left + 6, bottom + 110), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Age Group: "+str(profile[5]), (left + 6, bottom + 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Expression: "+text, (left + 6, bottom + 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Confidence: {:.2%}".format(confidence[0]), (left + 6, bottom + 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Time: "+Time, (left + 6, bottom + 230), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

            '''
            if profile[0] == 0:

                location="D:/Face/images2/0_{}.jpg".format(count)

                if np.max(confidence[0]):
                    
                    cv2.imwrite(location, roii)
                    count += 1

            if profile[0] == 1:

                unique_count = len(data["id"][1])
                
                cv2.putText(img, "Count: "+str(unique_count), (left + 6, bottom + 260), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                  
                filename = "Counts/counter.csv"
                  
                # writing to csv file 
                with open(filename, 'a', newline='') as csvfile:

                    fieldnames = ["Count", "Time"]
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'Count': unique_count, 'Time': Time})

                df = pd.read_csv("Counts/counter.csv")

                total = df["Count"].count()

                print("Total number of instances: ",total)

                count = 1
                location="D:/Face/images2/{}.jpg".format(count)
                cv2.imwrite(location, roii)
                
                for filename in glob.glob(location):

                    if Image.open(filename):

                        time_skips = float(time)

                        with video_capture(VIDEO_PATH) as cap:

                            cap.set(cv2.CAP_PROP_POS_MSEC, (countt*time_skips))

                            location="D:/Face/images2/1_{}.jpg".format(countt)
                            cv2.imwrite(location, roii)
                            countt += 1
                
            if profile[0] == 2:

                count = 2
                location="C:/Users/Aditya/Desktop/face/images2/{}.jpg".format(count)
                cv2.imwrite(location, roii)
            '''

            Id = profile[0]
            age_group = profile[5]
            gender = profile[3]

            insertdata(Id, age, gender, "_", Time, text, age_group, confidence[0], "_")


        cv2.rectangle(img, ((0,img.shape[0] -25)),(300, img.shape[0]), (255,255,255), -1)

        cv2.putText(img, "Number of persons detected: {}".format(len(face_locations)), (0,img.shape[0] -8), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (255,0,0), 1)
            
        cv2.putText(img, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (320, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Recognize", img)


        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(1)

        if key == 27:
            break


#cap.release()
cv2.destroyAllWindows()
