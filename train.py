import numpy as np
from PIL import Image
import os
import cv2

def train_classifier(data_dir):
    path = [os.path.join (data_dir,f) for f in os.listdir(data_dir)]
    
    faces = []
    ids = []
    prevLabel = ""
    labelFile  = open("label.txt", "a+")
    readFile = open("label.txt", "r").read()
    print(readFile)
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
        label = os.path.split(image)[1].split('.')[0]
        faces.append(imageNp)
        ids.append(id)
        
        if (label != prevLabel) and (label not in readFile):
            labelFile.write("\n{0}.{1}".format(id, label))
            print(label)
            prevLabel = label
    
    ids = np.array(ids)
    #print(labels)
    #print(ids)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    labelFile.close()

train_classifier("data")
print("Done Training!")
        
        