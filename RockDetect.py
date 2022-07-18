import os
from datetime import datetime
import numpy as np
import time

import cv2

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 0
rec = 0

# make shots directory to save pics
try:
    os.mkdir('./capture')
except OSError as error:
    pass

# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
#                                './saved_model/res10_300x300_ssd_iter_140000.caffemodel')
net = None
camera = cv2.VideoCapture(0)

class RockDetect:
    # video record
    def record(self):
        global rec_frame
        while rec:
            time.sleep(0.05)
            self.write(rec_frame)

    # record rocks and rock fractures
    def detect_rocks_fractures(self):
        global net
        (h, w) = self.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(self, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        confidence = detections[0, 0, 0, 2]
        if confidence < 0.5:
            return self
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        try:
            self = self[startY:endY, startX:endX]
            (h, w) = self.shape[:2]
            r = 480 / float(h)
            dim = (int(w * r), 480)
            self = cv2.resize(self, dim)
        except Exception as e:
            pass
        return self

    def gen_frames(self):
        # generate frame by frame from camera
        global out, capture, rec_frame
        while True:
            success, frame = camera.read()
            if success:
                if face:
                    frame = self.detect_face(frame)
                if grey:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if neg:
                    frame = cv2.bitwise_not(frame)
                if capture:
                    capture = 0
                    now = datetime.datetime.now()
                    p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                    cv2.imwrite(p, frame)
                if (rec):
                    rec_frame = frame
                    frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 4)
                    frame = cv2.flip(frame, 1)
                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                    frame = buffer.tobytes()
                    yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                except Exception as e:
                    pass
            else:
                pass


