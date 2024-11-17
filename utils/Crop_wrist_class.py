import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 손목에 해당하는 클래스만 필터링
WRIST_CLASSES = [
    'Trapezium', 'Trapezoid', 'Capitate', 'Hamate',
    'Scaphoid', 'Lunate', 'Triquetrum', 'Pisiform'
]

def load_json(json_path):
    """JSON 파일을 로드하여 반환"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_wrist_mask(image_shape, annotations):
    """손목 클래스에 해당하는 마스크 생성"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for annotation in annotations['annotations']:
        if annotation['label'] in WRIST_CLASSES:
            points = np.array(annotation['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    return mask

def filter_wrist_annotations(annotations):
    """손목 클래스만 필터링한 새로운 어노테이션 반환"""
    wrist_annotations = []
    for annotation in annotations['annotations']:
        if annotation['label'] in WRIST_CLASSES:
            wrist_annotations.append(annotation)
    return wrist_annotations

def calculate_center_from_mask(mask):
    """마스크에서 중심점 계산"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    return center_x, center_y

def crop_and_resize_image(image, mask, center_x, center_y):
    """마스크를 기준으로 512x512 크롭 후 2048x2048 리사이즈"""
    crop_size = 512
    resize_size = 2048
    h, w = image.shape[:2]
    
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w, center_x + crop_size // 2)
    y2 = min(h, center_y + crop_size // 2)
    
     # 마스크 적용: 원본 이미지에서 마스크 영역만 남기기
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 마스크와 이미지를 크롭
    cropped_image = masked_image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    # 크롭된 마스크와 이미지를 리사이즈
    resized_image = cv2.resize(cropped_image, (resize_size, resize_size))
    return resized_image, cropped_mask, x1, y1, crop_size

def adjust_points(points, x_offset, y_offset, crop_size):
    """크롭 및 리사이즈에 따라 어노테이션 좌표 조정"""
    scale_factor = 2048 / crop_size
    adjusted_points = []
    for x, y in points:
        new_x = (x - x_offset) * scale_factor
        new_y = (y - y_offset) * scale_factor
        # 유효한 좌표만 추가 (리사이즈된 이미지 범위 내)
        if 0 <= new_x < 2048 and 0 <= new_y < 2048:
            adjusted_points.append([int(new_x), int(new_y)])
    return adjusted_points

def process_images_and_json(source_dir, json_dir, dcm_save_dir, json_save_dir):
    """이미지와 JSON을 처리하여 손목 어노테이션만 포함된 새로운 JSON 생성"""
    os.makedirs(dcm_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)
    
    for folder in tqdm(os.listdir(source_dir)):
        if not folder.startswith('ID'):
            continue
        
        try:
            folder_id = int(folder[2:])
        except ValueError:
            continue
        
        if folder_id < 1 or folder_id > 348:
            continue
        
        folder_path = os.path.join(source_dir, folder)
        json_folder_path = os.path.join(json_dir, folder)
        
        if not os.path.isdir(folder_path) or not os.path.isdir(json_folder_path):
            continue
        
        images = sorted(os.listdir(folder_path))
        json_files = sorted(os.listdir(json_folder_path))
        
        for image_file, json_file in zip(images, json_files):
            image_path = os.path.join(folder_path, image_file)
            json_path = os.path.join(json_folder_path, json_file)
            
            image = cv2.imread(image_path)
            annotations = load_json(json_path)
            
            # 손목 클래스만 필터링
            wrist_annotations = filter_wrist_annotations(annotations)
            if not wrist_annotations:
                continue
            
            # 마스크 생성 및 중심점 계산
            mask = create_wrist_mask(image.shape, {'annotations': wrist_annotations})
            center_x, center_y = calculate_center_from_mask(mask)
            
            if center_x is None or center_y is None:
                continue
            
            # 이미지 및 마스크 크롭 및 리사이즈
            resized_image, cropped_mask, x_offset, y_offset, crop_size = crop_and_resize_image(image, mask, center_x, center_y)
            
            # 어노테이션 좌표 조정
            new_annotations = []
            for annotation in wrist_annotations:
                adjusted_points = adjust_points(annotation['points'], x_offset, y_offset, crop_size)
                new_annotations.append({
                    'id': annotation['id'],
                    'type': annotation['type'],
                    'label': annotation['label'],
                    'points': adjusted_points
                })
            
            annotations['annotations'] = new_annotations
            
            # 저장할 폴더 생성
            dcm_save_folder = os.path.join(dcm_save_dir, folder)
            json_save_folder = os.path.join(json_save_dir, folder)
            os.makedirs(dcm_save_folder, exist_ok=True)
            os.makedirs(json_save_folder, exist_ok=True)
            
            # 이미지 저장
            new_image_name = image_file.replace('image', 'image1')
            save_image_path = os.path.join(dcm_save_folder, new_image_name)
            cv2.imwrite(save_image_path, resized_image)
            
            # JSON 저장
            new_json_name = json_file.replace('image', 'image1')
            save_json_path = os.path.join(json_save_folder, new_json_name)
            with open(save_json_path, 'w') as f:
                json.dump(annotations, f, indent=4)
            
            print(f"Processed and saved: {save_image_path} & {save_json_path}")

# 실행 예시
source_dir = '/data/ephemeral/home/data/train/DCM'
json_dir = '/data/ephemeral/home/data/train/outputs_json'
dcm_save_dir = '/data/ephemeral/home/data/train/DCM'
json_save_dir = '/data/ephemeral/home/data/train/outputs_json'

process_images_and_json(source_dir, json_dir, dcm_save_dir, json_save_dir)
