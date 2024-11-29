import pandas as pd

# csv 파일의 특정 클래스 결과값 교체

def replace_classes(change_path, base_path, output_path):
   """
   새로운 예측 결과에서 특정 클래스들만 기존 CSV에서 가져와서 교체
   원본 순서 유지
   """
   # 교체할 클래스 정의
   target_classes = ['Trapezoid', 'Pisiform']
   
   # CSV 파일 읽기
   change_df = pd.read_csv(change_path)  # 가져올 데이터
   base_df = pd.read_csv(base_path)           # 베이스가 될 데이터
   
   print("\n=== Before Replacement ===")
   print("New predictions class counts:")
   print(change_df['class'].value_counts())
   
   # 데이터프레임을 딕셔너리로 변환하여 빠른 검색 가능하게 함
   original_dict = base_df[base_df['class'].isin(target_classes)].set_index(['image_name', 'class'])['rle'].to_dict()
   
   # 새로운 데이터프레임 순회하면서 target_classes만 교체
   for idx in change_df.index:
       img_name = change_df.loc[idx, 'image_name']
       class_name = change_df.loc[idx, 'class']
       
       if class_name in target_classes:
           # 기존 데이터에서 해당하는 RLE 값 찾기
           key = (img_name, class_name)
           if key in original_dict:
               change_df.loc[idx, 'rle'] = original_dict[key]
   
   # 결과 저장
   change_df.to_csv(output_path, index=False)
   
   print("\n=== After Replacement ===")
   print(f"Total predictions: {len(change_df)}")
   print("\nFinal class counts:")
   print(change_df['class'].value_counts())
   print(f"\nOutput saved to: {output_path}")
   
   # 검증
   replaced_counts = change_df[change_df['class'].isin(target_classes)]['class'].value_counts()
   print("\n=== Replaced Classes Count ===")
   print(replaced_counts)

if __name__ == "__main__":
   # 파일 경로 설정
   change_csv = "교체할 특정 Class 결과값이 있는 파일 경로"
   base_csv = "나머지 클래스들이 있는 파일 경로"
   output_csv = "final_output3.csv"  # 최종 결과 저장할 파일
   
   try:
       replace_classes(change_csv, base_csv, output_csv)
   except FileNotFoundError as e:
       print(f"Error: {e}")
       print("Please check if the input CSV files exist and the paths are correct.")
   except Exception as e:
       print(f"An error occurred: {e}")