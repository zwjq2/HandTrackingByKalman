# 开发作者   ：Tian.Z.L
# 开发时间   ：2022/4/26  9:52
# 文件名称   ：HandTrackingByKalman.PY
# 开发工具   ：PyCharm
import numpy as np
import mediapipe as mp
import cv2

print(cv2.useOptimized())
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
# 定义x的初始状态 -- 需要修改，初始化为捕获手势的位置x坐标
x_mat = np.mat([[0, ], [0, ]])
y_mat = np.mat([[0, ], [0, ]])
# 定义初始状态协方差矩阵
p_x_mat = np.mat([[1, 0], [0, 1]])
p_y_mat = np.mat([[1, 0], [0, 1]])
# 定义初始化控制矩阵
b_mat = np.mat([[0.5, ], [1, ]])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([[1, 1], [0, 1]])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.03, 0], [0, 0.03]])
# 定义观测矩阵
h_mat = np.mat([1, 1])
# 定义观测噪声协方差
r_mat = np.mat([1])

video = cv2.VideoCapture(0)
first_frame_flag = True
while True:
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 9:
                    if first_frame_flag:
                        x_mat[0, 0] = cx
                        y_mat[0, 0] = cy
                        first_frame_flag = False
                    else:
                        x_predict = f_mat * x_mat
                        y_predict = f_mat * y_mat
                        p_x_predict = f_mat * p_x_mat * f_mat.T + q_mat
                        p_y_predict = f_mat * p_y_mat * f_mat.T + q_mat
                        kalman_x = p_x_predict * h_mat.T / (h_mat * p_x_predict * h_mat.T + r_mat)
                        kalman_y = p_y_predict * h_mat.T / (h_mat * p_y_predict * h_mat.T + r_mat)
                        x_mat = x_predict + kalman_x * (cx - h_mat * x_predict)
                        y_mat = y_predict + kalman_y * (cy - h_mat * y_predict)
                        p_x_mat = (np.eye(x_mat.shape[0]) - kalman_x * h_mat) * p_x_predict
                        p_y_mat = (np.eye(y_mat.shape[0]) - kalman_y * h_mat) * p_y_predict
                        noise_x = cx + int(np.random.normal(0, 1) * 10)
                        noise_y = cy + int(np.random.normal(0, 1) * 10)
                        cv2.circle(frame, (noise_x, noise_y), 6, (0, 0, 255),cv2.FILLED)
                        # cv2.circle(frame, (int(x_mat[0, 0]), int(y_mat[0, 0])), 3, (0, 255, 0), cv2.FILLED)
                        cv2.circle(frame, (int(x_mat[0, 0]), int(y_mat[0, 0])), 6, (0, 255, 255), cv2.FILLED)
                    break
    cv2.imshow('video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()

