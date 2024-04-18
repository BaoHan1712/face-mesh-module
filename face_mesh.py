import cv2 
import time
import mediapipe as mp

cap =cv2.VideoCapture(0)
# Set giảm độ phân giải để tăng tốc độ xử lý
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# vẽ lại các khớp mat
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_face= 1,minDetectionCon=0.5, minTrackCon=0.5)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius =1)
 # Chỉ xử lý mỗi 2 khung hình một lần
frame_skip = 2 
frame_count = 0

# tính FPS
pTime =0
cTime =0

while True:
    success,img=cap.read()
# chuyển màu từ BGR sang RGB
    imRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    frame_count = 1   
# xử lý ảnh để phát hiện mat
    results = faceMesh.process(imRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
# Vẽ ra các khớp nối
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACE_CONNECTIONS,drawSpec)
    
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Runtime", img)
# Thoát khỏi vòng lặp 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
