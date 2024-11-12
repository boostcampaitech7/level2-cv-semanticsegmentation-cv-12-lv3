import wandb

def set_wandb(configs):
    wandb.login(key=configs['api_key'])
    wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=configs['experiment_detail'],
        config={
                'model': configs['model_name'],
                'resize': configs['image_size'],
                'batch_size': configs['train_batch_size'],
                'learning_rate': configs['lr'],
                'epoch': configs['max_epoch']
            }
    )