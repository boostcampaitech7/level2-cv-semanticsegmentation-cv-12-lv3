from .base_loss import CustomBCEWithLogitsLoss
from .dice_loss import CustomDiceLoss
from .jaccard_loss import CustomJaccardLoss
from .focal_loss import CustomFocalLoss
from .tversky_loss import CustomTverskyLoss
from .combined_loss import CombinedLoss

class LossSelector():
    """
    loss를 새롭게 추가하기 위한 방법
        1. loss 폴더 내부에 사용하고자하는 custom loss 구현
        2. 구현한 Loss Class를 loss_selector.py 내부로 import
        3. self.loss_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 loss_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.loss_classes = {
            "BCEWithLogitsLoss" : CustomBCEWithLogitsLoss,
            "DiceLoss": CustomDiceLoss,
            "JaccardLoss": CustomJaccardLoss,
            "FocalLoss": CustomFocalLoss,
            "TverskyLoss": CustomTverskyLoss
        }

    def get_loss(self, loss_name, **loss_parameter):
        # Combined loss인 경우
        if loss_name == "Combined":
            losses = []
            weights = []
            
            # cfg.losses에서 설정 가져오기
            for loss_config in loss_parameter.get('losses', []):
                loss_fn = self.loss_classes.get(loss_config['name'])(**loss_config.get('params', {}))
                if loss_fn is not None:
                    losses.append(loss_fn)
                    weights.append(loss_config.get('weight', 1.0))
            
            return CombinedLoss(losses, weights)
        return self.loss_classes.get(loss_name, None)(**loss_parameter)