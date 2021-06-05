import cv2
import sys
import os
from random import randint

if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    print('Erro ao carregar os arquivos do Goturn')
    sys.exit()

tracker = cv2.TrackerGOTURN_create()

video = cv2.VideoCapture('videos/race.mp4')
if not video.isOpened():
    print('Vídeo não carregado')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Não foi possível ler o arquivo')
    sys.exit()

bbox = cv2.selectROI(frame, False)
#  print(bbox)

ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    timer = cv2.getTickCount()
    ok, bbox = tracker.update(frame)
    #   print(ok, bbox)

    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, 'Falha no rastreamento', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, 'Tracker Goturn', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
    cv2.putText(frame, 'FPS: '+str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break





