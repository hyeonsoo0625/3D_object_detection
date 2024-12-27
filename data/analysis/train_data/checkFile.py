import os
import json
import pandas as pd
import cv2
import matplotlib.pyplot as plt

BASE_DIR = '../../3d-object-detection-for-autonomous-vehicles/'

def check_file(file):
    file_path = os.path.join(BASE_DIR, 'train_data', f'{file}.json')

    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df.head()

def visualize_image(file_path):
    image = cv2.imread(os.path.join(BASE_DIR, 'train_images', file_path))
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 8))
    plt.imshow(image)
    plt.title(file_path)
    plt.axis('off')
    plt.show()
