import cv2

camera_pas_recup : bool = True

cap = None
print('cap au début',cap)
while camera_pas_recup : 
    try: 
        cap = cv2.VideoCapture("/dev/video1")
        ret, frame = cap.read()
        if cap and ret :
            camera_pas_recup = False
    except Exception as exp : 
        print("exception capturée : ",str(exp))
        continue


print("cap :",cap)
while True:
    ret, frame = cap.read()
    if not ret:
        #print("Frame non lue")
        continue

    cv2.imshow("camera virtuelle", frame)

    if cv2.waitKey(1) == 27:
        break
