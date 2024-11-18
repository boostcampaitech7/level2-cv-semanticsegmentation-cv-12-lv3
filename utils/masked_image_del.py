import os

def delete_augmented_images(source_dir, prefix="masked_image"):
    """
    주어진 source_dir에서 특정 prefix로 시작하는 파일을 삭제하는 함수.
    """
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue

        # 각 ID 폴더 내의 이미지 파일 검색
        for filename in os.listdir(folder_path):
            # 지정된 prefix로 시작하는 파일을 찾습니다.
            if filename.startswith(prefix) and filename.endswith('.png'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    os.remove(file_path)  # 파일 삭제
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# 원본 데이터 폴더 경로 설정
source_image_dir = '/data/ephemeral/home/data/train/DCM'

# 추가된 데이터 삭제
delete_augmented_images(source_image_dir)
