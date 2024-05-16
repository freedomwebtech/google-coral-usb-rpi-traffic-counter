import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import*
from vidgear.gears import CamGear


model = YOLO('best_full_integer_quant_edgetpu.tflite')



#cap = cv2.VideoCapture(0)
stream = CamGear(source='https://youtu.be/fgqtvt839RE', stream_mode = True, logging=True).start() # YouTube Video URL as input

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")



tracker=Tracker()
cy1=305
offset=8
list1=[]
while True:
    frame = stream.read()
    if frame is None:
        break
    frame=cv2.resize(frame,(800,600))


    results = model.predict(frame,imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        if cy1<(cy+offset) and cy1>(cy-offset):
           cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
           cvzone.putTextRect(frame, f'{id}', (cx, cy), 1, 1)
           cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
           if list1.count(id)==0:
              list1.append(id)

    # Display FPS on frame
    counter=(len(list1))    
    cv2.line(frame,(1,305),(799,305),(0,255,0),2)
    cvzone.putTextRect(frame,f'Counter{counter}',(50,60),2,2)
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()
