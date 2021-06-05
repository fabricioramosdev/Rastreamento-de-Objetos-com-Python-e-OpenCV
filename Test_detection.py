import cv2
imagem = cv2.imread('imagens/pessoas.jpg')
detector = cv2.CascadeClassifier('cascade/fullbody.xml')

image_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#  cv2.imshow("Pessoas", image_gray)

detections = detector.detectMultiScale(image_gray)

for (x, y, w, h) in detections:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2, 0)

cv2.imshow('Detections', imagem)
cv2.waitKey(0)
cv2.destroyWindow()
