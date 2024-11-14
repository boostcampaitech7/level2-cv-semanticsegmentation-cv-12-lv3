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

def download_ckpt_from_wandb(experiment_detail, checkpoint_name):
    ## 현재 임시로 미사용. 팀원과 회의 필요 및 구현 확인 안 해봄
    # Wandb 작업 생성
    api = wandb.Api()

    # Wandb 아티팩트 검색
    artifact = api.artifact(f"{experiment_detail}:best-Dice")
    artifact_dir = artifact.download(path_prefix=checkpoint_name)
    return artifact_dir

# TODO: 아티팩트에 모델 검증 결과 이미지 테이블 추가