import matplotlib.pyplot as plt
import cv2
img = plt.imread(
    "group_photo.jpg"
)

model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

faces = model.detectMultiScale(img)

for face in faces:
    x, y, w, h = face

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

plt.imshow(img)
