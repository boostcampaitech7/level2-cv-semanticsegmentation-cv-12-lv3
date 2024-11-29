import os
import cv2
import json
from tqdm import tqdm
import albumentations as A

# 특정 Augmentation을 적용한 이미지 생성하는 기능

# Augmentation 정의
transform = A.Compose([
    A.HorizontalFlip(p=1),
    A.Emboss(alpha=(0.5, 0.9), strength=(1.0, 1.5), p=1),   # 엠보싱 효과
    A.CLAHE(clip_limit=[2.0, 2.0], tile_grid_size=(8, 8), p=1.0),
])

def augment_and_save_images(source_dir):
    """
    주어진 source_dir의 이미지를 변형하여 dest_dir에 저장하는 함수.
    """
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 각 ID 폴더 내의 이미지 파일 검색
        for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
            if filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                
                # 이미지 파일 읽기
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error reading {file_path}")
                    continue
                
                # Augmentation 적용
                augmented = transform(image=image)['image']
                
                # 새로운 파일명 생성 ('image' 뒤에 '1' 추가)
                if filename.startswith("image"):
                    new_filename = filename.replace("image", "trasformed_image", 1)
                else:
                    new_filename = "1" + filename
                
                new_file_path = os.path.join(folder_path, new_filename)
                
                # 변형된 이미지 저장
                cv2.imwrite(new_file_path, augmented)
                print(f"Saved transformed image: {new_file_path}")



def flip_points_horizontally(points, width):
    """
    좌우 반전을 위해 x 좌표를 반전합니다.
    """
    return [[width - x, y] for x, y in points]

def process_json_files(source_dir, image_width):
    """
    주어진 source_dir의 JSON 파일을 읽어 변형된 points 좌표를 새로운 JSON으로 저장합니다.
    """
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # JSON 파일 변환
        for filename in tqdm(os.listdir(folder_path), desc=f"Processing JSON {folder}"):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 좌우 반전된 points 좌표 생성
                for annotation in data['annotations']:
                    annotation['points'] = flip_points_horizontally(annotation['points'], image_width)
                
                # 새로운 파일명 생성 ('image' 뒤에 '1' 추가)
                if filename.startswith("image"):
                    new_filename = filename.replace("image", "trasformed_image", 1)
                else:
                    new_filename = "1" + filename
                
                new_file_path = os.path.join(folder_path, new_filename)
                
                # 변형된 JSON 저장
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f"Saved transformed JSON: {new_file_path}")

# 원본 데이터 폴더 경로 설정
source_image_dir = '/data/ephemeral/home/data/train/DCM'
source_json_dir = '/data/ephemeral/home/data/train/outputs_json'
image_width = 2048  # 이미지의 너비

# Augmentation 적용 (이미지 및 JSON 파일)
augment_and_save_images(source_image_dir)
process_json_files(source_json_dir, image_width)