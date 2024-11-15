import os
import os.path as osp
import re, shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from dataset import XRayDataset

def set_wandb(configs):
    wandb.login(key=configs['api_key'])
    wandb_run = wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=configs['experiment_detail'],
        config={
                'model': configs['model_name'],
                'resize': configs['image_size'],
                'batch_size': configs['train_batch_size'],
                'loss_name': configs['loss_name'],
                'scheduler_name': configs['scheduler_name'],
                'learning_rate': configs['lr'],
                'epoch': configs['max_epoch']
            }
    )

    return wandb_run

def upload_ckpt_to_wandb(wandb_run, checkpoint_path):
    # Wandb 아티팩트 정의(아티팩트=모델, 데이터셋, 테이블 등의 잡동사니)
    # yaml config에서 experiment_detail로 설정한 이름으로 이름 설정
    artifact = wandb.Artifact(name=wandb_run.name, type="model")

    # 아티팩트에 모델 체크포인트 추가
    artifact.add_file(local_path=checkpoint_path, name='models/'+osp.basename(checkpoint_path))

    # 아티팩트를 Wandb에 저장
    wandb_run.log_artifact(artifact, aliases=["best-Dice"])

def download_ckpt_from_wandb(experiment_detail, checkpoint_name):
    ## 현재 임시로 미사용. 팀원과 회의 필요 및 구현 확인 안 해봄
    # Wandb 작업 생성
    api = wandb.Api()

    # Wandb 아티팩트 검색
    artifact = api.artifact(f"{experiment_detail}:best-Dice")
    artifact_dir = artifact.download(path_prefix=checkpoint_name)
    return artifact_dir

def wandb_table_after_evaluation(wandb_run, model, thr=0.5):
    # 클래스와 클래스 ID 정의
    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna', 'BG'
    ]   
    IDS = list(range(len(CLASSES)-1))+[255]
    image_root = "/data/ephemeral/home/data/train/DCM"
    label_root = "/data/ephemeral/home/data/train/outputs_json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 완디비에 마스크를 기록하는 랩핑 함수
    def wb_mask(bg_img, class_set, pred_mask=[], true_mask=[]):
        masks = {}
        if len(pred_mask) > 0:
            masks["prediction"] = {"mask_data" : pred_mask}
        if len(true_mask) > 0:
            masks["ground truth"] = {"mask_data" : true_mask}
        return wandb.Image(bg_img, classes=class_set, masks=masks)

    def extract_hand(filename):
        """
        파일 이름에서 왼손(L) 또는 오른손(R)을 추출하는 함수

        Args:
            filename: 파일 이름

        Returns:
            str: 파일 ID와 추출된 손(L 또는 R)
        """
        match = re.search(r"(.+?)/(.+?)/(.+?)_([LR])\.png", filename)
        if match:
            return f"{match.group(2)}_{match.group(4)}"
        else:
            return None

    # 완디비 클래스 오브젝트를 설정하여 시각화에 메타데이터 추가
    class_set = wandb.Classes([{'name': name, 'id': id} 
                            for name, id in zip(CLASSES, IDS)])
    
    # 아티팩트(=버전 있는 폴더) 생성
    artifact = wandb.Artifact(name=wandb_run.name, type="table")

    # 데이터셋을 올릴 완디비 테이블 오브젝트 설정
    columns=["file_name",
             "prediction",
             "ground_truth", 
        # "AvgDiceScore"
        ]
    table = wandb.Table(
        columns=columns
    )

    # 중간 시각화 결과물을 저장할 임시 폴더
    TMPDIR = "tmp_labels"
    if not os.path.isdir(TMPDIR):
        os.mkdir(TMPDIR)

    # 고정된 특정 이미지 셋을 배치로
    # 왼손, 오른손 별로 나와서 2배로 나옵니다
    outlier_ids = ["ID073", "ID288", "ID363", 
                #    "ID364", "ID387",
        # "ID430", "ID487", "ID506", "ID523", "ID543"
        ]
    fnames = sorted([
        osp.relpath(osp.join(root, fname), start=image_root)
        for root, _, files in os.walk(image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png" \
        and any(root.endswith(f"/{id}") for id in outlier_ids)
    ])
    labels = sorted([
        osp.relpath(osp.join(root, fname), start=label_root)
        for root, _, files in os.walk(label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json" \
        and any(root.endswith(f"/{id}") for id in outlier_ids)
    ])
    dataset = XRayDataset(fnames=fnames,
                       labels=labels,
                       image_root=image_root,
                       label_root=label_root,
                       fold=None,
                       transforms=[],
                       is_train=False,
                       )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print("Start adding inference to wandb table...")
    with tqdm(total=len(data_loader), desc="[Adding inference to table]", disable=False) as pbar:
        # 추론 및 테이블 채우기
        for i, batch in enumerate(data_loader):
            # 원본 이미지 배열
            img, mask = batch
            bg_image = (img[0].permute(1, 2, 0)*255).cpu().numpy().astype(np.uint8)
            img = img.to(device)

            # 모델의 예측 값
            output = model(img)

            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            prediction_tf = (output > thr)[0].detach().cpu().numpy()
            mask_tf = (mask > thr)[0].detach().cpu().numpy()
            prediction = np.ones((2048, 2048), dtype=int)*255
            ground_truth = np.ones((2048, 2048), dtype=int)*255
            for j in range(mask_tf.shape[0]):
                prediction[prediction_tf[j]]=j
                ground_truth[mask_tf[j]]=j

            # Dice 계산
            # dice = dice_coef(prediction, mask)
            # avg_dice = torch.mean(dice, dim=?).item()

            # 최종 데이터 열 추가
            row = [
                extract_hand(fnames[i]),
                wb_mask(bg_image, class_set, pred_mask=prediction),
                wb_mask(bg_image, class_set, true_mask=ground_truth),
            ]  # row.extend로 클래스별 dice도 기록 가능
            table.add_data(*row)
            pbar.update(1)
        
    # 아티팩트에 테이블 추가
    artifact.add(table, "val_prediction_result")
        
    # 마지막으로, 아티팩트 기록
    print("Saving data to WandB...")
    wandb_run.log_artifact(artifact)
    wandb_run.finish()
    print("... Run Complete")

    # 임시 폴더와 파일 삭제
    shutil.rmtree(TMPDIR)