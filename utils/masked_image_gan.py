import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

# Masked 이미지 생성

CLASS = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IDX = {v: i + 1 for i, v in enumerate(CLASS)}

def apply_mask_and_save_images_and_json(source_dir, json_dir, target_classes=[i for i in range(1, 30)]):
    """
    JSON 파일의 어노테이션 정보를 기반으로 마스크를 생성하여 이미지에 적용하고,
    새로운 JSON 파일을 생성하여 저장하는 함수.
    """
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        json_folder_path = os.path.join(json_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue

        # Process each image in the folder
        for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
            if filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                json_path = os.path.join(json_folder_path, filename.replace('.png', '.json'))

                # Load the original image
                image = cv2.imread(file_path)
                image = image / 255.0
                image = image.astype("float32")
                original_image = image.copy()

                if not os.path.exists(json_path):
                    print(f"JSON file not found: {json_path}")
                    continue

                # Load the label file
                with open(json_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    annotations = data["annotations"]

                # Create an empty mask for the specific classes
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                new_annotations = []

                # Iterate over annotations and create a mask for the target classes
                for ann in annotations:
                    target_c = ann["label"]
                    target_c = CLASS2IDX[target_c]

                    # Only include the specified target classes
                    if target_c in target_classes:
                        points = np.array(ann["points"], np.int32)

                        # Create a polygon mask for the specific class
                        mask_img = Image.new('L', (image.shape[1], image.shape[0]), 0)
                        ImageDraw.Draw(mask_img).polygon([tuple(point) for point in points], outline=1, fill=1)
                        class_mask = np.array(mask_img, dtype=np.uint8)

                        # Combine with the overall mask
                        mask = np.maximum(mask, class_mask)
                        new_annotations.append(ann)  # Add the annotation to the new list

                # Apply the mask to the original image
                masked_image = original_image.copy()
                masked_image[mask == 0] = 0  # Set the background to black

                # Save the masked image
                new_filename = filename.replace("image", "masked_image", 1)
                new_file_path = os.path.join(folder_path, new_filename)
                cv2.imwrite(new_file_path, (masked_image * 255).astype(np.uint8))
                print(f"Saved masked image: {new_file_path}")

                # Create a new JSON file with the updated annotations
                new_json_filename = filename.replace("image", "masked_image", 1).replace('.png', '.json')
                new_json_path = os.path.join(json_folder_path, new_json_filename)

                new_data = {
                    "annotations": new_annotations
                }

                # Save the new JSON file
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=4)
                print(f"Saved new JSON: {new_json_path}")

# 원본 이미지와 JSON 폴더 경로 설정
source_image_dir = '/data/ephemeral/home/data/train/DCM'
source_json_dir = '/data/ephemeral/home/data/train/outputs_json'

# JSON 파일의 어노테이션 정보를 기반으로 마스크 적용 및 이미지와 JSON 파일 저장
apply_mask_and_save_images_and_json(source_image_dir, source_json_dir)