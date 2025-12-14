import cv2
video=cv2.VideoCapture("./videos/video3.mp4")
num=0
save_step=30
while True:
    ret, frame=video.read()
    if not ret:
        break
    num+=1
    if num%save_step==0:
        cv2.imwrite("./images/images3/images"+str(num)+".jpg",frame)