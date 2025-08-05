# Efficient Motion Prediction (EMP) with Trajectory Distance Distillation

## Setup
install requirements from py12.11_requirements.txt


## Model Names
1. emp_small
2. emp_smal_thin
3. emp_tiny
4. emp_tiny_thin

## Run training
### Baseline (No distillation and no Teacher model)
python train.py data_root=data model={name of model} batch_size=48 monitor=val_minFDE6 model.target.decoder=detr
### Non-Baseline (Provide the teacher mode)
python train.py data_root=data model={name of model} batch_size=48 +model.target.teacher_weights=checkpoints/empd.ckpt monitor=val_minFDE6 model.target.decoder=detr

## Run evaluation
python eval.py data_root=data batch_size=32 'checkpoint="{path to last checkpoint of trained model}"' model.target.decoder=detr