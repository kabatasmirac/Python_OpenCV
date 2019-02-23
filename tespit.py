import cv2, os
tanimlayici = cv2.face.LBPHFaceRecognizer_create()
tanimlayici.read('ogrenilen_veriler_image_formatinda/ogrenilen_veriler.yml')
cascadePath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'taninan_veriler_image_formatinda'

cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()
    if ret is True:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        image_path = [os.path.join(path, f) for f in os.listdir(path)]
        for (x, y, w, h) in faces:

            tahminEdilenKisi, conf = tanimlayici.predict(gray[y:y+h,x:x+w])
            dosya= "taninan_veriler_image_formatinda/" + str(tahminEdilenKisi) + "/"
            tahminEdilenKisi=(os.listdir(dosya)[0].split(".")[0])
            print(conf)
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.75
            fontColor = (127, 0, 63)
            cv2.putText(im, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
            print(tahminEdilenKisi)
            cv2.imshow('im', im)
            cv2.waitKey(1000)