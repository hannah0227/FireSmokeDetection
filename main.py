!nvidia-smi

import os
WORK_DIR = '/content/drive/MyDrive/YOLOv10'
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

!git clone https://github.com/THU-MIG/yolov10.git #yolov10 clone하기

os.chdir('yolov10')

!pip install -r requirements.txt -q
!pip install -e . -q

!wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt

!pip install roboflow #roboflow에서 데이터셋 불러옴

from roboflow import Roboflow
rf = Roboflow(api_key="YXi61oUHqbwrUMKuIHnC")
project = rf.workspace("catargiuconstantin").project("firesmokedataset")
version = project.version(2)
dataset = version.download("yolov5")

!yolo task=detect mode=train \
      model='yolov10s.pt' \
      data= '/content/drive/MyDrive/YOLOv10/yolov10/FireSmokeDataset-2/data.yaml' \
      epochs=20 \
      imgsz=640 \
      batch=16 \
      patience=10 \
      optimizer='SGD'

import cv2
from ultralytics.models.yolov10 import YOLOv10
import numpy as np
from google.colab.patches import cv2_imshow # Colab에서 이미지를 표시하기 위해 import

# --- 헬퍼 함수: 두 박스가 겹치는지 확인 (IoU 기준) ---
def boxes_overlap(box1, box2, threshold=0.01):
    # box: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return False

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area

    return iou > threshold

# --- 헬퍼 함수: 겹치는 박스들을 그룹으로 묶고 병합 ---
def merge_overlapping_boxes(boxes):
    if not boxes:
        return []

    groups = [[b] for b in boxes]

    while True:
        merged_in_pass = False
        new_groups = []
        used_indices = set()

        for i in range(len(groups)):
            if i in used_indices:
                continue

            current_group = list(groups[i]) # Make a mutable copy

            for j in range(i + 1, len(groups)):
                if j in used_indices:
                    continue

                other_group = groups[j]

                is_overlapping = any(boxes_overlap(b1, b2) for b1 in current_group for b2 in other_group)

                if is_overlapping:
                    current_group.extend(other_group)
                    used_indices.add(j)
                    merged_in_pass = True

            new_groups.append(current_group)

        groups = new_groups
        if not merged_in_pass:
            break

    final_boxes = []
    for group in groups:
        group_boxes = np.array(group)
        min_x = np.min(group_boxes[:, 0])
        min_y = np.min(group_boxes[:, 1])
        max_x = np.max(group_boxes[:, 2])
        max_y = np.max(group_boxes[:, 3])
        final_boxes.append([min_x, min_y, max_x, max_y])

    return final_boxes

# ----------------- 메인 코드 -----------------

MODEL_PATH = '/content/drive/MyDrive/YOLOv10/yolov10/runs/detect/train56/weights/best.pt'
model = YOLOv10(MODEL_PATH)

VIDEO_PATH = '/content/drive/MyDrive/YOLOv10/yolov10/FireSmokeDataset-2-pp/fire_video.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

FIRE_CLASS_INDEX = 0
SMOKE_CLASS_INDEX = 2

# 결과 동영상 저장을 위한 설정
output_dir = '/content/drive/MyDrive/YOLOv10/yolov10/runs/detect/predictions_video'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'output_yolov10n_fin.mp4') # 결과 파일명 변경
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

time_stamps = []
fire_area_data = []
smoke_area_data = []

print(f"동영상 처리를 시작합니다: {VIDEO_PATH}")
print(f"결과는 다음 경로에 저장됩니다: {output_video_path}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)

    fire_boxes = []
    smoke_boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        if class_id == FIRE_CLASS_INDEX:
            fire_boxes.append(box.xyxy[0].cpu().numpy().tolist())
        elif class_id == SMOKE_CLASS_INDEX:
            smoke_boxes.append(box.xyxy[0].cpu().numpy().tolist())

    # 겹치는 불꽃(fire) 박스들을 병합
    merged_fire_boxes = merge_overlapping_boxes(fire_boxes)
    # 겹치는 연기(smoke) 박스들을 병합
    merged_smoke_boxes = merge_overlapping_boxes(smoke_boxes)

    # 불꽃(fire) 영역 계산 및 그리기 (빨간색)
    total_fire_area = 0
    for box in merged_fire_boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        total_fire_area += area
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)

    # 연기(smoke) 영역 계산 및 그리기 (회색)
    total_smoke_area = 0
    for box in merged_smoke_boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        total_smoke_area += area
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (128, 128, 128), 3)


    # 화면에 총 면적 표시
    cv2.putText(frame, f"Total Fire Area: {int(total_fire_area)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Smoke Area: {int(total_smoke_area)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    time_stamps.append(frame_count / 24)
    fire_area_data.append(total_fire_area)
    smoke_area_data.append(total_smoke_area)

    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}...")

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print("\n동영상 처리가 완료되었습니다!")
print(f"결과 영상은 {output_video_path} 에서 확인하실 수 있습니다.")

import pandas as pd
import matplotlib.pyplot as plt

# --- [그래프 보정 추가] 이동 평균을 적용하는 부분 ---

# 1. 이동 평균을 계산할 윈도우 크기 설정 (이 값을 조절하며 부드러운 정도를 찾으세요)
#    (예: 24fps 영상에서 1초간의 평균을 보려면 window_size=24)
WINDOW_SIZE = 12

# 2. Pandas Series로 데이터 변환
fire_series = pd.Series(fire_area_data)
smoke_series = pd.Series(smoke_area_data)

# 3. 이동 평균 계산
#    min_periods=1 옵션은 데이터 시작 부분에서 윈도우 크기보다 데이터가 적어도 평균을 계산하게 해줍니다.
smoothed_fire_data = fire_series.rolling(window=WINDOW_SIZE, min_periods=1).mean()
smoothed_smoke_data = smoke_series.rolling(window=WINDOW_SIZE, min_periods=1).mean()


# --- 그래프 생성 및 출력 (보정된 데이터 사용) ---

plt.figure(figsize=(14, 7)) # 그래프 크기 조정

# 원본 데이터를 얇고 투명하게 그려서 비교용으로 표시 (선택 사항)
plt.plot(time_stamps, fire_area_data, color='red', alpha=0.3, linestyle='--', label='Original Fire Area')
plt.plot(time_stamps, smoke_area_data, color='gray', alpha=0.3, linestyle='--', label='Original Smoke Area')

# 이동 평균으로 보정된 데이터를 굵은 선으로 그리기
plt.plot(time_stamps, smoothed_fire_data, label='Smoothed Fire Area', color='red', linewidth=2.5)
plt.plot(time_stamps, smoothed_smoke_data, label='Smoothed Smoke Area', color='gray', linewidth=2.5)

# 그래프 제목 및 라벨 설정
plt.title('Smoothed Fire and Smoke Area Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Area (pixels²)')
plt.legend()
plt.grid(True)

# 그래프를 이미지 파일로 저장하고 화면에 출력
graph_path = os.path.join(output_dir, 'graph_yolov10n_fin.png')
plt.savefig(graph_path)
plt.show()

print(f"보정된 면적 변화 그래프는 {graph_path} 에 저장되었습니다.")
