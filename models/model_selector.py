from .base_model import UnetModel, UnetPlusPlus
from .swin_unet import SwinUnet
from .unet_transform import R2U_Net, AttU_Net, R2AttU_Net
from .effisegnet import EffiSegNetBN
from .mask2former import Mask2Former
from .segformer import SegFormerModel
from .upernet import UperNet
class ModelSelector():
    """
    model을 새롭게 추가하기 위한 방법
        1. model 폴더 내부에 사용하고자하는 custom model 구현
        2. 구현한 Model Class를 model_selector.py 내부로 import
        3. self.model_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 model_name을 설정한 key값으로 변경

    * Swin Model 같은 경우 pretrain model이 필요하여 초기에는 pretrain_ckpt에 있는 가중치 사용 *
    * 최종 prediction이 나오면 해당 pt로 경로 설정 (swin_unet.py에서) *
    """
    def __init__(self) -> None:
        self.model_classes = {
            "Unet" : UnetModel,
            "R2U_Net" : R2U_Net,
            "AttU_Net" : AttU_Net,
            "R2AttU_Net" : R2AttU_Net,
            "UnetPlusPlus": UnetPlusPlus,
            "Swin" : SwinUnet,
            "EffiSegNet": EffiSegNetBN,
            "UperNet": UperNet,
            "Mask2Former": Mask2Former,
            "SegFormer": SegFormerModel
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)