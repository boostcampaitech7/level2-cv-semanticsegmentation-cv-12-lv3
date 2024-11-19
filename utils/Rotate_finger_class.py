import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 손가락에 해당하는 클래스
FINGER_CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium', 'Trapezoid', 'Capitate', 'Hamate',
    'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform'
]

def load_json(json_path):
    """JSON 파일을 로드하여 반환"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_mask(image_shape, annotations):
    """손가락 클래스에 해당하는 마스크를 생성"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for annotation in annotations['annotations']:
        label = annotation['label']
        if label in FINGER_CLASSES:
            points = np.array(annotation['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    return mask

def rotate_point(x, y, cx, cy, angle):
    """좌표를 주어진 각도로 회전"""
    rad = np.deg2rad(-angle)
    new_x = (x - cx) * np.cos(rad) - (y - cy) * np.sin(rad) + cx
    new_y = (x - cx) * np.sin(rad) + (y - cy) * np.cos(rad) + cy
    return int(new_x), int(new_y)

def rotate_annotations(annotations, image_shape, angle):
    """JSON 파일의 손가락 클래스 좌표를 회전"""
    (h, w) = image_shape[:2]
    cx, cy = w // 2, h // 2  # 이미지 중심점
    
    rotated_annotations = []
    for annotation in annotations['annotations']:
        label = annotation['label']
        if label in FINGER_CLASSES:
            rotated_points = []
            for point in annotation['points']:
                x, y = point
                new_x, new_y = rotate_point(x, y, cx, cy, angle)
                rotated_points.append([new_x, new_y])
            
            rotated_annotations.append({
                'id': annotation['id'],
                'type': annotation['type'],
                'label': label,
                'points': rotated_points
            })
    return rotated_annotations

def rotate_image(image, angle):
    """이미지를 주어진 각도로 회전"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

def process_images_and_json(source_dir, json_dir, dcm_save_dir, json_save_dir, max_id=348):
    os.makedirs(dcm_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)
    
    for folder in tqdm(os.listdir(source_dir)):
        if not folder.startswith('ID'):
            continue
        
        try:
            folder_id = int(folder[2:])
        except ValueError:
            continue
        
        if folder_id < max_id:
            continue
        
        folder_path = os.path.join(source_dir, folder)
        json_folder_path = os.path.join(json_dir, folder)
        
        if not os.path.isdir(folder_path) or not os.path.isdir(json_folder_path):
            continue
        
        images = sorted(os.listdir(folder_path))
        json_files = sorted(os.listdir(json_folder_path))
        
        for idx, (image_file, json_file) in enumerate(zip(images, json_files)):
            image_path = os.path.join(folder_path, image_file)
            json_path = os.path.join(json_folder_path, json_file)
            
            # JSON 파일 로드
            annotations = load_json(json_path)
            
            # 원본 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # 마스크 생성
            mask = create_mask(image.shape, annotations)
            
            # 마스크를 이미지에 적용
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            
            # 회전 각도 설정
            angle = -45 if idx == 0 else 45
            
            # 이미지 회전
            rotated_image = rotate_image(masked_image, angle)
            
            # JSON 어노테이션 좌표 회전
            rotated_annotations = rotate_annotations(annotations, image.shape, angle)
            annotations['annotations'] = rotated_annotations
            
            # 회전된 이미지 저장 (DCM 폴더)
            new_image_name = image_file.replace('image', 'masked_image')
            dcm_save_folder = os.path.join(dcm_save_dir, folder)
            os.makedirs(dcm_save_folder, exist_ok=True)
            save_image_path = os.path.join(dcm_save_folder, new_image_name)
            cv2.imwrite(save_image_path, rotated_image)
            
            # 회전된 JSON 저장 (outputs_json 폴더)
            new_json_name = json_file.replace('image', 'masked_image')
            json_save_folder = os.path.join(json_save_dir, folder)
            os.makedirs(json_save_folder, exist_ok=True)
            save_json_path = os.path.join(json_save_folder, new_json_name)
            
            with open(save_json_path, 'w') as f:
                json.dump(annotations, f, indent=4)
            
            print(f"Saved: {save_image_path} and {save_json_path}")

# 실행 
source_dir = '/data/ephemeral/home/data/train/DCM'
json_dir = '/data/ephemeral/home/data/train/outputs_json'
dcm_save_dir = '/data/ephemeral/home/data/train/DCM'
json_save_dir = '/data/ephemeral/home/data/train/outputs_json'
process_images_and_json(source_dir, json_dir, dcm_save_dir, json_save_dir)
