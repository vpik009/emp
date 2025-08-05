The hint distillation trainer class is made with the exact same interface as the existing EMP trainer. Thus to train a model, simply use:

train.py model=small_thin epochs=75

The model argument specifies the student model. Our options are: small_thin, tiny_thin, small_wide, tiny_wide.

15 epochs are reserved for hint-based pre-training. Thus the above command does 60 training epochs of the whole student model.

Likewise, evaluation uses the provided eval.py script. Specify the checkpoint created during training.
