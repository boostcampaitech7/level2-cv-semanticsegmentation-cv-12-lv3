# 📋 Project Overview


![project_image](https://github.com/user-attachments/assets/8b7c2877-efae-4e3b-895e-8415705ac748)

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

1. 질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

2. 수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

3. 의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

4. 의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

**Input**

- hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.

**Output**

- 모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당합니다.

최종적으로 예측된 결과를 Run-Length Encoding(RLE) 형식으로 변환하여 csv 파일로 제출합니다.

<br/>

# 🗃️ Dataset
![image](https://github.com/user-attachments/assets/c2a3a918-6594-4493-8d39-b9c1d0cd8202)
- 이미지 크기
  - **(2048,2048), 3 channel**

- 양손을 촬영 했기 때문에 사람 별로 두 장의 이미지가 존재

<br/>

# 😄 Team Member

<!-- <table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/655258d1-43fd-4db7-a3fb-0d5a37c45774" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/1ec21af7-8c04-4e99-9922-4275e56516c4" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/f8ce149b-06dd-466b-ba16-83523e3f1abe" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/406da993-6556-4238-ab22-74239c870aaa" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/3bedb72c-bf6b-4feb-b486-3232e2363406" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/86ce3850-aa0a-4564-ba35-65c1af08c85f" width="140" height="140"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/lkl4502" target="_blank">오홍석</a></td>
        <td><a href="https://github.com/lexxsh" target="_blank">이상혁</a></td>
        <td><a href="https://github.com/yejin-s9" target="_blank">이예진</a></td>
        <td><a href="https://github.com/Haneol-Kijm" target="_blank">김한얼</a></td>
        <td><a href="https://github.com/PGSammy" target="_blank">조재만</a></td>
        <td><a href="https://github.com/oweixx" target="_blank">방민혁</a></td>
    </tr>
    <tr align="center">
        <td>T7208</td>
        <td>T7221</td>
        <td>T7225</td>
        <td>T7138</td>
        <td>T7253</td>
        <td>T7158</td>
    </tr>
</table> -->
<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/lkl4502">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003880%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>오홍석</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lexxsh">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003955%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이상혁</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/yejin-s9">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003894%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이예진</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Haneol-Kijm">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003885%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김한얼</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PGSammy">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003890%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>조재만</b></sub><br />
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/oweixx">
        <img src="https://stages.ai/_next/image?url=https%3A%2F%2Faistages-api-public-prod.s3.amazonaws.com%2Fapp%2FUsers%2F00003872%2Fuser_image.png&w=1920&q=75" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>방민혁</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>

<br />

## Project Structure
```
📦level2-cv-12
 ┣ 📂configs
 ┃ ┗ 📜base_train.yaml
 ┃
 ┣ 📂loss
 ┃ ┣ 📜base_loss.py
 ┃ ┣ 📜combined_loss.py
 ┃ ┣ 📜dice_loss.py
 ┃ ┣ 📜focal_loss.py
 ┃ ┣ 📜jaccard_loss.py
 ┃ ┣ 📜tversky_loss.py
 ┃ ┗ 📜loss_selector.py
 ┃
 ┣ 📂models
 ┃ ┣ 📜base_model.py
 ┃ ┣ 📜model_selector.py
 ┃ ┣ 📜effisegnet.py
 ┃ ┣ 📜mask2former.py  
 ┃ ┣ 📜segformer.py
 ┃ ┣ 📜swin_unet.py
 ┃ ┣ 📜swin_unet_base.py
 ┃ ┣ 📜unet_transform.py
 ┃ ┣ 📜upernet.py         
 ┃ ┗ 📜model_selector.py
 ┃
 ┣ 📂scheduler
 ┃ ┗ 📜scheduler_selector.py
 ┃
 ┣ 📂utils
 ┃ ┣ 📜Crop_wrist_class.py
 ┃ ┣ 📜Offline_augmentation.py
 ┃ ┣ 📜Rotate_finger_class.py
 ┃ ┣ 📜change_class.py
 ┃ ┣ 📜hard_voting_ensemble.ipynb
 ┃ ┣ 📜masked_image_del.py
 ┃ ┣ 📜masked_image_gan.py
 ┃ ┣ 📜notion.py
 ┃ ┣ 📜soft_voting.py
 ┃ ┗ 📜wandb.py
 ┃
 ┣ 📂EDA
 ┃ ┣ 📜EDA.ipynb
 ┃ ┣ 📜Output_Visualization.ipynb
 ┃ ┣ 📜Transform.ipynb 
 ┃ ┣ 📜Test_data_angle_analysis.ipynb  
 ┃ ┗ 📜Masked_Visualization.ipynb
 ┃
 ┣ 📜train.py
 ┣ 📜trainer.py                      
 ┣ 📜inference.py                   
 ┣ 📜dataset.py             
 ┗ 📜README.md
```

## Commit Convention
1. `Feature` ✨ **새로운 기능 추가**
2. `Bug` 🐛 **버그 수정**
3. `Docs` 📝 **문서 수정**
4. `Refactor` ♻️ **코드 리펙토링**

커밋할 때 헤더에 위 내용을 작성하고 전반적인 내용을 간단하게 작성합니다.

### 예시

- `git commit -m "[#이슈번호] ✨ feat 간단하게 설명" `
- `git commit -m "[#이슈번호] 🐛 bug 간단하게 설명"`
- `git commit -m "[#이슈번호] 📝 docs 간단하게 설명" `
- `git commit -m "[#이슈번호] ♻️ refactor 간단하게 설명" `

<br/>

## Branch Naming Convention

브랜치를 새롭게 만들 때, 브랜치 이름은 항상 위 `Commit Convention`의 Header와 함께 작성되어야 합니다.

### 예시

- `Feature/~~~`
- `Refactor/~~~`

## Offline Augmentation 적용

1. Masked
- 기존 Train의 img와 GT가 담긴 JSON을 이용하여 각 클래스 부분만 Masking된 img를 Offline Augmentation으로 추가하였습니다.
![image](https://github.com/user-attachments/assets/529d822e-a0e4-494a-8ace-c4ac4950fb1b)

2. Rotate, Crop
- 이미 손목이 돌아가 있는 데이터를 제외하고 finger 부분만 Masking하여 40~50도 Rotate 적용시킨 img를 Offline Augmentation으로 추가하였습니다.
![image](https://github.com/user-attachments/assets/d0e2434d-c0b7-4be3-a8f8-0cd5f873e933)

3. 클래스 특화
- Masked를 적용한 img에서 finger를 제외한 손등뼈부분 중 val점수가 비교적 낮은 trapzoid, pisiform 두가지 클래스만 GT로 설정하여 Offline Augmentation으로 추가했습니다.
![image](https://github.com/user-attachments/assets/c5f44439-54a7-4b19-a047-402c3d6e9241)

## Final Structure












## 최종 순위
