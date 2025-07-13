import cv2
from ultralytics import YOLO

video_path=0
capture=cv2.VideoCapture(video_path)

KEYPOINTS_NAMES=[
    'nose',
    'eye(L)',
    'eye(R)',
    'ear(L)',
    'ear(R)',
    'sholder(L)',
    'sholder(R)',
    'elbow(L)',
    'elbow(R)',
    'wrist(L)',
    'wrist(R)',
    'hip(L)',
    'hip(R)',
    'knee(L)',
    'knee(R)',
    'abkle(L)',
    'ankle(R)',
]

model=YOLO("yolov8n-pose.pt")

def process_frame(frame):
    
    results=model(frame)
    annotated_frame=results[0].plot()

    #keypoints=results[0].keypoints.xy
    #confs=results[0].keypoints.conf

    #for keypoint in keypoints:
        #for idx, point in enumerate(keypoint):
            #x,y=int(point[0]),int(point[1])
            #score=confs[0][idx]

            #if score <0.5:
                #continue

    keypoints = results[0].keypoints.xy[0]  # 1人目のキーポイント
    confs = results[0].keypoints.conf[0]   # 1人目の信頼度

    for idx, (point, score) in enumerate(zip(keypoints, confs)):
        x, y = int(point[0]), int(point[1])
        if score < 0.5:
            continue
        
        left_raised =False 
        right_raised =False
        both_raised = False

        right_shoulder=keypoints[6]
        right_elbow = keypoints [8]
        right_wrist = keypoints[10]

        left_shoulder=keypoints[5]
        left_elbow = keypoints [7]
        left_wrist = keypoints[9]

        
        if right_wrist[1]<right_elbow[1] and right_elbow[1] < right_shoulder[1] and left_wrist[1]<left_elbow[1] and left_elbow[1] < left_shoulder[1]:
            both_raised=True
        elif right_wrist[1]<right_elbow[1] and right_elbow[1] < right_shoulder[1]:
            right_raised=True
        elif left_wrist[1]<left_elbow[1] and left_elbow[1] < left_shoulder[1]:
            left_raised=True

        if right_raised:
            cv2.putText(
                annotated_frame,
                'right Hand Raised',
                (50,50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.3,
                color=(255,140,0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        if left_raised:
            cv2.putText(
                annotated_frame,
                'Left Hand Raised',
                (50,50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.3,
                color=(0,255,0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
        if both_raised:
             cv2.putText(
                annotated_frame,
                'both Hand Raised',
                (50,50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.3,
                color=(0,0,255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            

    
        

    print(f"Keypoint Name={KEYPOINTS_NAMES[idx]},X={x},Y={y},Score={score:.4f}")

    cv2.rectangle(
        annotated_frame,
        (x,y),
        (x+3,y+3),
        (255,0,255),
        cv2.FILLED,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_frame,
        KEYPOINTS_NAMES[idx],
        (x+5,y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,0,255),
        1,
        cv2.LINE_AA,
    )

    print ("------------------------------------------------------")
    return annotated_frame 

while capture.isOpened():
    success, frame =capture.read()
    if not success:
        break

    annotated_frame = process_frame(frame)
    cv2.imshow("YOLOv8 Human Pose Estimation",annotated_frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

capture.release()
cv2.destroyAllWindows()