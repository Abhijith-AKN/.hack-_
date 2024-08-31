#chaio
import cv2
import numpy as np
import pyttsx3

#text-to-speech
engine = pyttsx3.init()

#Using YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

#setting the network to use CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Using COCO class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

    
#video capturing
cap = cv2.VideoCapture(0)

#resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Processing frame
frame_skip = 2

while True:
    ret, frame = cap.read()

    #Checking
    if not ret:
        print("Failed to capture frame")
        break

    #Skip frames
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_skip != 0:
        continue

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #processing yolo
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = set()  #set to track detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.add(classes[class_id])

    #Applying non-max suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #bounding boxes
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  #Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    #voice command
    for obj in detected_objects:
        engine.say(f"I see a {obj}")
        engine.runAndWait()

    #Display video
    cv2.imshow('webcam', frame)

    #q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
