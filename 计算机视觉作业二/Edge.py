import cv2


# https://blog.csdn.net/weixin_40922285/article/details/102967331

cap = cv2.VideoCapture(1)
ret, frame = cap.read()


for i in range(100):
    
    ret, frame = cap.read()
    
    frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edge = cv2.Canny(frame, 10, 30)
    
    cv2.imshow('Canny Edge', edge)
    cv2.imshow('input', frame)
    
    
    if cv2.waitKey(20) == ord('q'):
        break