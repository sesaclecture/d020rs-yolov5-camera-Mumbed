import torch
import cv2
import os

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
while True:
    # Read frame (BGR to RGB)
    ret, frame = cap.read()
    # TODO: break the loop on error
    if not ret:
        break
    # 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        # TODO: 인식결과를 표시하기 위한 좌표를 얻음
        x1, y1, x2, y2, _, cls = map(int, obj[:6])
        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        conf = obj[4]
        obj_info = list(map(int, obj))
        print()
        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        cv2.rectangle(frame, pt1= (x1,y1),pt2= (x2, y2),color=(0,255,0),thickness=3)
        cv2.putText(frame, model.names[obj_info[5]],(x1,y1-10), cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)
        cv2.putText(frame, str(conf),(x1,y1-20), cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),3)

        print(f"Object {i}: {model.names[obj_info[5]]}")

    # TODO: 화면 표시
    
    cv2.imshow("test", frame)
    # TODO: 종료를 위한 key 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
cap.release()
cv2.destroyAllWindows()
