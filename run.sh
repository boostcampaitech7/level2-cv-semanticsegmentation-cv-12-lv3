#! bin/bash

python train.py --config /data/ephemeral/home/lexxsh/level2-cv-semanticsegmentation-cv-12-lv3/configs/eff_v2.yaml
python inference.py ./checkpoints/Unet/eff_v2/best* --output output/eff_v2_output.csv --resize 2048

python train.py --config /data/ephemeral/home/lexxsh/level2-cv-semanticsegmentation-cv-12-lv3/configs/vgg.yaml
python inference.py ./checkpoints/Unet/vgg/best* --output output/vgg_output.csv --resize 2048
