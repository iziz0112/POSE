import cv2
from ultralytics import YOLO

video_path=0
capture=cv2.VideoCapture(video_path)

KEYPOINTS_NAMES=[
    "nose",         # 0 鼻
    "eye(L)",       # 1 左目
    "eye(R)",       # 2 右目
    "ear(L)",       # 3 左耳
    "ear(R)",       # 4 右耳
    "shoulder(L)",  # 5 左肩
    "shoulder(R)",  # 6 右肩
    "elbow(L)",     # 7 左肘
    "elbow(R)",     # 8 右肘
    "wrist(L)",     # 9 左手首
    "wrist(R)",     # 10 右手首
    "hip(L)",       # 11 左腰
    "hip(R)",       # 12 右腰
    "knee(L)",      # 13 左膝
    "knee(R)",      # 14 右膝
    "ankle(L)",     # 15 左足首
    "ankle(R)",     # 16 右足首
]

def hen(Ax,Bx,Ay,By):                           #二点の座標の[x,y]をそれぞれ計４つ、入力するとその二点の距離がわかる関数
    henn=((Bx-Ax)**2+(By-Ay)**2)**0.5
    return henn

def cos(a,b,c):                                 #cosがわかる関数                
    coss=(a**2 + b**2 - c**2)/(2*a*b)
    return coss

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

        left_hip=keypoints[11]          #left_hipは「キーポイント11（左腰）[x,y]」
        left_knee=keypoints[13]         #left_kneeは「キーポイント13（左ひざ）[x,y]」
        left_ankle=keypoints[15]        #left_ankleは「キーポイント15（左足首）[x,y]」

        left_knee_raised=False          #left_knee_raisedをFalseにする。(今後使う。)

        hipL__kneeL=hen(left_hip[0],left_knee[0],left_hip[1],left_knee[1])          #先ほど定義したhen関数を使って、左腰→左ひざの長さ
        kneeL__ankleL=hen(left_knee[0],left_ankle[0],left_knee[1],left_ankle[1])    #ひざ→足首の長さ
        hipL__ankleL=hen(left_hip[0],left_ankle[0],left_hip[1],left_ankle[1])       #腰→足首の長さ

        kneeL_cos=cos(hipL__kneeL,kneeL__ankleL,hipL__ankleL)       #左ひざの角度（cos）


        if kneeL_cos >=-0.65:                   #大体５０°(cos-0.65)膝を上げたら判定開始。（180゜<=>150゜以上になったら発動。）
            left_knee_raised=True               #left_knee_raisedをTrueにする。

        if left_knee_raised:                    #left_knee_raisedがTrueになり発動。             
            cv2.putText(
                annotated_frame,
                'Left knee Raised',
                (50,50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.3,
                color=(0,255,0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )                                   #テキストの色や内容、場所などを定義。
       
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