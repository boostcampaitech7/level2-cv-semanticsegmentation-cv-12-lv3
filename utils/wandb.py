import wandb

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

def wandb_table_after_evaluation(wandb_run, model, num_examples, thr=0.5):
    # 클래스와 클래스 ID 정의
    from PIL import Image
    CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]   
    IDS = list(range(len(CLASSES)))

    # 완디비에 마스크를 기록하는 랩핑 함수
    def wb_mask(bg_img, pred_mask=[], true_mask=[]):
        masks = {}
        if len(pred_mask) > 0:
            masks["prediction"] = {"mask_data" : pred_mask}
        if len(true_mask) > 0:
            masks["ground truth"] = {"mask_data" : true_mask}
        return wandb.Image(bg_img, classes=class_set, masks=masks)

    # 완디비 클래스 오브젝트를 설정하여 시각화에 메타데이터 추가
    class_set = wandb.Classes([{'name': name, 'id': id} 
                            for name, id in zip(CLASSES, IDS)])
    
    # 아티팩트(=버전 있는 폴더) 생성
    artifact = wandb.Artifact(name=wandb_run.name, type="table")

    # 데이터셋을 올릴 완디비 테이블 오브젝트 설정
    columns=["id", "prediction", "ground_truth", "AvgDiceScore"]
    table = wandb.Table(
        columns=columns
    )

    # 중간 시각화 결과물을 저장할 임시 폴더
    TMPDIR = "tmp_labels"
    if not os.path.isdir(TMPDIR):
        os.mkdir(TMPDIR)    

    # 고정된 특정 이미지 셋을 배치로
    specific_image_path = ["path"]
    ??
    batch = []
    for img, label in zip(img_list, label_list):
        batch.append((img, label))
    batch = [Image]


    # 추론 및 테이블 채우기
    for i, img, mask in enumerate(batch):
        # 원본 이미지 배열
        orig_image = img[0]???
        bg_image = image2np(orig_image.data*255).astype(np.uint8)

        # 모델의 예측 값
        output = model(img)

        output = F.interpolate(output, size=(mask_h, mask_w), mode="bilinear")
        output = torch.sigmoid(output)
        prediction = (output > thr).detach().cpu().numpy()
        
        # Dice 계산
        dice = dice_coef(prediction, mask)
        # avg_dice = torch.mean(dice, dim=?).item()

        # 최종 데이터 열 추가
        row = [
            str(batch_ids[i]), 
            wb_mask(bg_image, pred_mask=prediction),
            wb_mask(bg_image, true_mask=mask),
            avg_dice
        ]  # row.extend로 클래스별 dice도 기록 가능
        table.add_data(*row)
        
    # 아티팩트에 테이블 추가
    artifact.add(table, "val_prediction_result")
        
    # 마지막으로, 아티팩트 기록
    print("Saving data to WandB...")
    run.log_artifact(artifact)
    run.finish()
    print("... Run Complete")

    # 임시 폴더와 파일 삭제
    shutil.rmtree(TMPDIR)