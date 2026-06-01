import cv2
import os
import numpy as np
from glob import glob
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt

model = deeplabv3_resnet101(pretrained=True).eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((520,520)), #모델입력크기(고정값)
    T.ToTensor(),
])

#pascalvoc 데이터의 라벨종류
LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]

img_paths = glob(os.path.join('images','*.png'))
for img_path in img_paths:
    img_orig = cv2.imread(img_path)

    print(f"이미지크기 : {img_orig.shape}")

    # cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
    # cv2.COLOR_BGR2RGB: 이미지 색상 채널을 변경 (BGR 형식을 RGB 형식으로 변경)
    # plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시, 입력은 RGB(A) 데이터 혹은 2D 스칼라 데이터
    # plt.show(): 현재 열려있는 모든 figure를 표시 (여기서 figure는 이미지, 그래프 등)
    #plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    #plt.show()

    input_tensor = transform(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # 원본 크기로 Resize
    output_predictions_resized = cv2.resize(output_predictions, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(output_predictions_resized, cmap="jet", alpha=0.7)
    # plt.title("Segmentation Mask (Resized)")
    # plt.show()

    unique_classes = np.unique(output_predictions_resized)
    unique_classes

    for class_id in unique_classes:
        print(LABEL_NAMES[class_id])

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    colormap[:20]

    seg_map = (output_predictions_resized == 8) #classid 8 :cat
    img_mask = seg_map.astype(np.uint8) *255
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

    # plt.imshow(img_mask, cmap='gray')
    # plt.show()

    img_show = cv2.addWeighted(img_orig, 0.6, color_mask, 0.4, 0.0)
    # plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    # plt.show()

    #배경 흐리게 하기

    img_orig_blur = cv2.blur(img_orig, (25, 25))
    # plt.imshow(cv2.cvtColor(img_orig_blur, cv2.COLOR_BGR2RGB))
    # plt.show()

    #이미지 색상변경
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    #이미지반전(배경0고양이255 -> 배경255고양이0)
    img_bg_mask = cv2.bitwise_not(img_mask_color)
    #배경만있는영상얻기
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)

    # plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
    # plt.show()

    # np.where(조건, 참일때, 거짓일때)
    # 마스크가 255면, 원본이미지 참, 블러인 이미지가 거짓
    img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
    plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
    plt.show()