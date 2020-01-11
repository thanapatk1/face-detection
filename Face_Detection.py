import cv2
import serial
import time

arduino_use = False

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = cv2.face.LBPHFaceRecognizer_create()
classifier.read("classifier.xml")
labelFile = open("label.txt", "r")
labels = labelFile.readlines()

if arduino_use:
    arduino = serial.Serial('COM4', 9600)

def draw_boundary(img,classifier,scaleFactor,minNighbors,color, clf):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNighbors)
        coords=[]
        for(x,y,w,h) in features:
            id,confidence = clf.predict(gray[y:y+h,x:x+w])
            for label in labels:
                if id == int(label.split(".")[0]):
                    if confidence < 60:
                        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        cv2.putText(img, "{0} : {1} %".format(label.split(".")[1], round(100 - confidence)),(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
                        if arduino_use:
                            arduino.write(b"o")
                    else :
                        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                        cv2.putText(img, "Unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
                        if arduino_use:
                            arduino.write(b"c")
            coords=[x,y,w,h]
        return img,coords
        
def detect(img,Cascade,clf):
        _img,coords = draw_boundary(img, Cascade, 1.1, 10, (0,0,255), clf)
        if len(coords) == 4:
                result = _img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
                
        return _img
        
cap = cv2.VideoCapture(0)

lastTime = time.time()
while True:
        ret,frame = cap.read()
        frame = detect(frame, faceCascade, classifier)
        fps = round(1/(time.time()-lastTime), 1)
        cv2.putText(frame, f"FPS : {str(fps)}",(0,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
        cv2.imshow("face detection", frame)
        lastTime = time.time()
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()
