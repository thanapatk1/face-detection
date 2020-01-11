import cv2
import numpy as np
from PIL import Image
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labelFile = open("label.txt", "a+")
readFile = open("label.txt", "r").read()
labels = open("label.txt", "r").readlines()

def create_dataset(img,id,img_id):
    cv2.imwrite("data/" + label + "." +str(id)+"."+str(img_id)+".jpg",img)
    
def draw_boundary(img,classifier,scaleFactor,minNighbors,color,text):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNighbors)
        coords=[]
        for(x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
                coords=[x,y,w,h]
        return img,coords
        
def detect(img,Cascade,img_id):
        _img,coords = draw_boundary(img, Cascade, 1.1, 10, (0,0,255), "Saving Face : {0} %".format(round((img_id / detectAmount* 100), 1)))
        if len(coords) == 4:
                id = int(labels[len(labels) - 1].split(".")[0]) + 1
                result = _img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
                create_dataset(result, id, img_id)
        return _img

def train_classifier(data_dir):
    path = [os.path.join (data_dir,f) for f in os.listdir(data_dir)]
    
    faces = []
    ids = []
    prevLabel = ""
    #print(readFile)
    
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split('.')[1])
        label = os.path.split(image)[1].split('.')[0]
        faces.append(imageNp)
        ids.append(id)
        
        if (label != prevLabel) and (label not in readFile):
            labelFile.write("\n{0}.{1}".format(id, label))
            #print(label)
            prevLabel = label
    
    ids = np.array(ids)
    #print(labels)
    #print(ids)
    
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    labelFile.close()
        
cap = cv2.VideoCapture(0)
i = 0
detectAmount = 500
label = input("Enter label : ")

while (i < detectAmount):
        ret,frame = cap.read()
        frame = detect(frame, faceCascade, i)
        i += 1
        cv2.imshow("face detection", frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()
print("Done Collecting Pictures!")
train_classifier("data")
print("Done Training!")


