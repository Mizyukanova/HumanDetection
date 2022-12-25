import numpy as np
import cv2 as cv

print("person detector")

cv.startWindowThread()

# open video stream from file
cap = cv.VideoCapture('dashcam.mp4')

while cap.isOpened():
    # capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resizing for faster detection
    frame1 = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2) ) 
    # converting an image from bgr color space to gray
    frame2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
  
    # cutting sky from frame
    new_frame1 = np.zeros(frame1.shape[:2], dtype='uint8')
    rectangle = cv.rectangle(new_frame1.copy(), (0, 130), (640, 360), 255, -1)
    frame3 = frame2
    frame4 = cv.bitwise_and(frame3, frame3, mask=rectangle)

    #initializing model of person detector  
    body = cv.CascadeClassifier('haarcascade_fullbody.xml')
    results = body.detectMultiScale(frame4, scaleFactor=1.1, minNeighbors=4)

    isDetected = False
    for (x, y, w, h) in results:
#       # display the detected people in the colour picture
        cv.rectangle(frame1, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        cv.putText(frame1, 'human', (x - 10, y - 10), cv.FONT_HERSHEY_PLAIN, 1, 
                          (0, 255, 0), 2)
        isDetected = True

    # video displaying
    cv.imshow('video', frame1)
    if isDetected:
        cv.waitKey(10000)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()