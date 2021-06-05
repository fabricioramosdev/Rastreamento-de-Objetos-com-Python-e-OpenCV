import cv2
import sys
from random import randint

tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture('videos/walking.avi')
if not video.isOpened():
    print('Vídeo não carregado')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Não foi possível ler o arquivo')
    sys.exit()

cascade = cv2.CascadeClassifier('cascade/fullbody.xml')

def detectar():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(frame_gray)

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('Detections', frame)
            #cv2.waitKey(0)
            #cv2.destroyWindow()
            if x > 0:
                print('Detecção efetuada pelo haardcascade')
                return x, y, w, h


bbox = detectar()


ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        print('Falha no rastramento. Será executado o detector haardcascade')
        bbox = detectar()
        tracker = cv2.TrackerMOSSE_create()
        ok = tracker.init(frame, bbox)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27:
        break
