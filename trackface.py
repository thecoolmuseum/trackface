from multiprocessing import Value, Array
import multiprocessing
import concurrent.futures
import time

import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import pickle


class GaussianFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.prev_value = None

    def filter(self, value):
        if self.prev_value is None:
            self.prev_value = value
            return value
        else:
            self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value 
            return self.prev_value
        
class AdaptiveGaussianFilter:
    def __init__(self, alpha_min, alpha_max, diff_min, diff_max):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.diff_min = diff_min
        self.diff_max = diff_max
        self.prev_value = None

    def filter(self, value):
        if self.prev_value is None:
            self.prev_value = value
            return value
        else:
            diff = np.linalg.norm(value - self.prev_value)
            alpha = map_and_trim(diff, self.diff_min, self.diff_max, self.alpha_min, self.alpha_max)
            self.prev_value = alpha * value + (1 - alpha) * self.prev_value
            self.prev_diff = diff
            return self.prev_value
            
class AlphaBetaFilter:
    def __init__(self, alpha, beta, dt):
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.x_hat = np.array([0.0])
        self.v_hat = np.array([0.0])

    def filter(self, x):
        if np.all(self.x_hat == 0) and np.all(self.v_hat == 0):
            self.x_hat = x
            self.v_hat = np.zeros_like(x)
            return x
        else:
            r = x - self.x_hat
            self.x_hat += self.v_hat * self.dt + self.alpha * r
            self.v_hat += (self.beta * r) / self.dt
            return self.x_hat

class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def filter(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
def map_and_trim(value, in_min, in_max, out_min, out_max):
    # 入力範囲に基づいて値を正規化
    value_normalized = (value - in_min) / (in_max - in_min)

    # 正規化された値を出力範囲にマップ
    value_mapped = value_normalized * (out_max - out_min) + out_min

    # 値を出力範囲内にトリム
    value_trimmed = max(min(value_mapped, out_max), out_min)

    return value_trimmed



def count_cameras():
    n = 0
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                break
            cap.release()
            n += 1
        except:
            break
    return n




def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        change_camera()
def change_camera():
    global cap, camera_index, camera_count
    # カメラのインデックスを切り替える
    cap.release()
    camera_index = (camera_index + 1) % camera_count
    cap = cv2.VideoCapture(camera_index)

# Processで動作するフレームループ
def frame_loop(tracking, target):
    fps = 60
    try:
        # filter = GaussianFilter(1 / fps)
        filter = AdaptiveGaussianFilter(0.5 / fps, 10 / fps, 30, 200)
        # filter = AlphaBetaFilter(0.05, 0.005, 0.03)
        # filter = MovingAverageFilter(10)
        
        print('tracking start')
        while tracking.value:
        # for _ in range(5):
            # print(f'tracking {tracking.value} {target}')
            filterd = filter.filter(np.array([target[0], target[1]]))
            x_pos = filterd[0]
            y_pos = filterd[1]

            pyautogui.moveTo(x_pos, y_pos)

            # 60fpsで待機
            time.sleep(1 / fps)
        print('tracking end')

    except Exception as e:
        print('error' + e)
        tracking.value = False


if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    camera_index = 0
    camera_count = count_cameras()
    cv2.namedWindow('MediaPipe FaceMesh')
    cv2.setMouseCallback('MediaPipe FaceMesh', mouse_callback)
    cap = cv2.VideoCapture(camera_index)
    mouth_open = False

    # データ共有用のマネージャー
    manager = multiprocessing.Manager()
    # トラッキングフラグ
    tracking = manager.Value('b', False)
    # ターゲット座標
    target = manager.list([0.0, 0.0])

    executor = None

    pyautogui.FAILSAFE = False

    max_fps = 10
    detected_fps = max_fps
    last_time = time.time() - 1 / max_fps

    try:
        while cap.isOpened():
            now = time.time()
            delta_time = now - last_time
            last_time = now

            if delta_time > 0:
                detected_fps = 1 / delta_time

            ret, frame = cap.read()
            if ret:
                # BGRからRGBに変換
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_image)
                # 塗りつぶし
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        for id, lm in enumerate(face_landmarks.landmark):
                            # 各ランドマークの座標を取得
                            h, w, c = frame.shape
                            x, y = int(lm.x * w), int(lm.y * h)
                            # ランドマークを描画
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
                            

                        # 口の開きが口の幅の20%以上のときに口が開いていると判定
                        # 口の幅を取得 
                        mouth_width = abs(face_landmarks.landmark[308].x - face_landmarks.landmark[78].x)
                        # 口の開きを取得
                        mouth_height = abs(face_landmarks.landmark[14].y - face_landmarks.landmark[13].y)
                        open = mouth_height > mouth_width * 0.2
                        close = mouth_height <= mouth_width * 0.1
                        # 口が閉じた状態から口が開いたときに中ボタンをダウン
                        if open and not mouth_open:
                            pyautogui.mouseDown(button='middle')
                            # print('mouth open')
                            mouth_open = True
                        # 口が開いた状態から口が閉じたときに中ボタンをアップ
                        if close and mouth_open:
                            pyautogui.mouseUp(button='middle')
                            # print('mouth close')
                            mouth_open = False

                        # 顔の向きを取得
                        face_nose = np.array([face_landmarks.landmark[6].x, face_landmarks.landmark[6].y, face_landmarks.landmark[6].z])
                        face_right = np.array([face_landmarks.landmark[127].x, face_landmarks.landmark[127].y, face_landmarks.landmark[127].z])
                        face_left = np.array([face_landmarks.landmark[356].x , face_landmarks.landmark[356].y, face_landmarks.landmark[356].z])
                        face_center = (face_right + face_left) / 2
                        face_front = face_nose - face_center
                        face_front_normarized = face_front / face_front[2]
                        x_pos = map_and_trim(face_front_normarized[0], -0.38, 0.38, 0, pyautogui.size()[0])
                        y_pos = map_and_trim(-face_front_normarized[1], -0.22, 0.22, 0, pyautogui.size()[1])
                        target[0]=x_pos
                        target[1]=y_pos

                        break

                # 左右反転
                frame = cv2.flip(frame, 1)
                # デバッグ表示
                cv2.putText(frame, f'Camera: {camera_index}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if results.multi_face_landmarks is not None: 
                    cv2.putText(frame, f'{face_nose}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'{face_center}', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'{face_front_normarized}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'{x_pos} {y_pos}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'{detected_fps:05.2f}fps {delta_time:05.2f}sec', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # 画面表示
                cv2.imshow('MediaPipe FaceMesh', frame)
                
            # キー入力
            key = cv2.waitKey(1)
            # Cでカメラ切り替え
            if key & 0xFF == ord('c'):
                change_camera()
            # spaceでトラッキング開始
            if key & 0xFF == 32:
                tracking.value = not tracking.value
                if tracking.value:
                    # Threadでフレームループを開始
                    if executor is not None:
                        executor.shutdown()
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
                    executor.submit(frame_loop, tracking, target)
                    print('tracking started')

            # esc or Qで終了
            if key & 0xFF == 27 or key & 0xFF == ord('q'):
                print('exit')
                break
            process_time = time.time() - now
            time.sleep(max(0, 1 / max_fps - process_time))
    except Exception as e:
        print(e)

    tracking.value = False

    if executor is not None:
        executor.shutdown()
    cap.release()
    cv2.destroyAllWindows()