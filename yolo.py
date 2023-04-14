import cv2 as cv
import numpy as np

net=cv.dnn.readNet('yolov3.weights','yolov3.cfg')

classes=[]
with open('coco.names','r') as f:
    classes=f.read().splitlines()
boxes=[]
confidences=[]
class_ids=[]
count=0
capture=cv.VideoCapture(r"C:\Users\Nitin V Kavya\Desktop\Yolo\funny.mp4")
while True:
    isTrue,img=capture.read()


    
    height,width,_=img.shape

    blob=cv.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)

    output_layers_names=net.getUnconnectedOutLayersNames()
    layerOutputs=net.forward(output_layers_names)
    



    for output in layerOutputs:
        for detection in output:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)

                x=int(center_x-w/2)
                y=int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes=cv.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font=cv.FONT_HERSHEY_COMPLEX
    colors=np.random.uniform(0,255,size=(len(boxes),3))

    for i in indexes.flatten():
        x,y,w,h=boxes[i]
        label=str(classes[class_ids[i]])
        confidence=str(round(confidences[i]))
        color=colors[i]
        cv.rectangle(img,(x,y),(x+w,y+h),color,thickness=3)
        cv.putText(img,label+ " "+confidence,(x,y+20),font,2,(255,255,255),2)
        if label=='person':
            crop=img[x:x+w,y:y+h]
            cv.imwrite("./captured"+"/"+str(count)+".png",crop)
            count=count+1

        




    cv.imshow('img',img)
    if cv.waitKey(20) & 0xFF==ord('b'):
        break
capture.release()
cv.destroyAllWindows()

