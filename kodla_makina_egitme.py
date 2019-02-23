#makinanın kişileri tanımasını sağlayacağız
import cv2,os
import numpy as np
from PIL import Image

tanimlama = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier("face.xml")
path = 'taninan_veriler_image_formatinda'

def get_images_and_labels(path: object) -> object:
    image_path = [os.path.join(path, f) for f in os.listdir(path)]
    resimler = []
    etiketler = []
    for i in range(len(os.listdir(path))):
        for image in os.listdir(image_path[i]):
            new_path = path + "/" + image_path[i].split("/")[1] +"/"+image
            image_pil = Image.open(new_path).convert('L')
            img = np.array(image_pil, "uint8")
            nbr = np.int_(os.path.split(image)[1].split("_")[0])
            print(nbr)

            faces = faceCascade.detectMultiScale(img, 1.1, 3)

            for (x, y, w, h) in faces:
                resimler.append(img[y: y + h, x: x + w])
                etiketler.append(np.int_(nbr))
                cv2.imshow("resim egitme setine ekleniyor...", img[y: y + h, x: x + w])
                cv2.waitKey(1)

    return resimler, etiketler

resimler, etiketler = get_images_and_labels(path)
#print(etiketler)## etiketlerde tc tutuluyor
#print(resimler)##matris
cv2.waitKey(1)
tanimlama.train(resimler, np.array(etiketler))
tanimlama.write('ogrenilen_veriler_image_formatinda/ogrenilen_veriler.yml')

cv2.destroyAllWindows()
