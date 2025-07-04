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

knee_y_history = []
count=0                                             #ここの二つは外に出しておかないと毎コンマ、リセットされる。

model=YOLO("yolov8n-pose.pt")                   

def process_frame(frame):
    
    results=model(frame)
    annotated_frame=results[0].plot()
    keypoints = results[0].keypoints.xy[0]  # 1人目のキーポイント
    confs = results[0].keypoints.conf[0]   # 1人目の信頼度
    global count                                   #最初に参照にすべきcountを用意（中で用意すると毎回リセット。グローバルをつけないと参照元を見失う）

    for idx, (point, score) in enumerate(zip(keypoints, confs)):
        x, y = int(point[0]), int(point[1])
        if score < 0.5:
            continue

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
        print(f"Keypoint Name={KEYPOINTS_NAMES[idx]},X={x},Y={y},Score={score:.4f}")
                   
        left_knee_y=keypoints[13][1]            #left_knee_yは「キーポイント１３（左ひざ）のy軸」
        knee_y_history.append(left_knee_y)      #左ひざの座標データをknee_y_historyに入れていく
         
        if len(knee_y_history) > 10:            #座標データが１０を超えたら
            knee_y_history.pop(0)               #リストの一番左（一番古いデータ）を取り出す　#appendは左から押し詰めていく。

                   
        
        if len(knee_y_history) >= 5:            #リストの中身が５個を超えたら判定開始。
            delta_y = knee_y_history[-5] - knee_y_history[-1]   #delta_y 　＝　0.5秒前のデータ　ー　現在の時間  （正確には違うが）      
            if delta_y > 20:                    #０．５秒前と今の座標に２０pxの差があれば
                count = 300                      #countを３００にする

        if count > 0:                           #countが0より多きれば起動。（↑で300になるので起動）
            cv2.putText(
                annotated_frame,
                'left knee Raised',
                (50,50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.3,
                color=(0,255,0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )                                   #色や場所、画面に表示されるテキストなどを設定。
            count -=1                           #countが減り、0になったら表示が終わる
       
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