import cv2,os

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('face.xml')
i=0
ad = input('Kişi ismi soyisim giriniz...("isim_soyisim" belirtilen formatta olmak zorunludur...)')
g_id = input('Güvenlik rakamı giriniz...')
kisi_id = g_id+"_"+ad

while True:
    _, img =cam.read()
    if _ is True:
        os.makedirs("taninan_veriler_image_formatinda/" + g_id + "/", exist_ok=True)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=0, minSize=(100, 100),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in faces:
            i=i+1

            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            cv2.imshow('resim', img[y:y + h, x:x + w])
            cv2.imwrite("taninan_veriler_image_formatinda/" + g_id + "/" + kisi_id +"."+str(i)+ ".jpg",
                            gray[y:y + h, x:x + w])
            cv2.waitKey(1000)


            if i > 15:
                cam.release()
                cv2.destroyAllWindows()
                break
