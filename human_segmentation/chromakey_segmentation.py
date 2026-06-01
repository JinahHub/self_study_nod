import cv2
import os
import numpy as np
from glob import glob
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt

#이미지 불러오기
cat_img_path = os.path.join('images','hoochoo_3.png')
human_img_path = os.path.join('images','jina.jpeg')

cat_img = cv2.imread(cat_img_path)
human_img = cv2.imread(human_img_path)

cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2RGB)
human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)

print(f"원본 고양이 이미지 크기: {cat_img.shape}")
print(f"배경 이미지 크기: {human_img.shape}")

model = deeplabv3_resnet101(pretrained=True).eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520, 520)),  # 모델 입력 크기
    T.ToTensor(),
])

input_tensor = transform(human_img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

print(f"추론 마스크 크기 (Before Resize): {output_predictions.shape}")
output_predictions_resized = cv2.resize(output_predictions, (human_img.shape[1], human_img.shape[0]), interpolation=cv2.INTER_NEAREST)
print(f"추론 마스크 크기 (After Resize): {output_predictions_resized.shape}")

unique_classes = np.unique(output_predictions_resized)
print(f"예측된 클래스 ID: {unique_classes}")

target_class_id = unique_classes[-1]

seg_map = (output_predictions_resized == target_class_id)
img_mask = seg_map.astype(np.uint8) * 255

# 가장 큰 덩어리(사람) 하나만 남기기
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_mask, connectivity=8)
# label 0은 배경이므로 제외하고, 면적(CC_STAT_AREA)이 가장 큰 라벨 선택
if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    img_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

# 배경을 고양이 이미지 크기에 맞춤
cat_img_resized = cv2.resize(cat_img, (human_img.shape[1], human_img.shape[0]))
# 고양이 부분만 남기고 배경 적용
img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)  # 3채널 변환
result_img = np.where(img_mask_color == 255, human_img, cat_img_resized)  # 마스크 기반 합성

plt.imshow(result_img)
plt.show()

#파일 저장 (RGB -> BGR 변환 후 저장)
output_path = os.path.join('images', 'result_img.png')
result_save = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, result_save)
print(f"결과 이미지 저장 완료: {output_path}")