import torch
from thop import profile
from pathlib import Path
from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from torch.utils.data import DataLoader
from src.model.trainer_forecast import Trainer


ckpt_path = "checkpoints/empd.ckpt"  # Kenta and Ole: change this to the model checkpoint
decoder_type = "detr"
device = "cpu"  # set this to cuda if you guys have a GPU

# loading the model
model = Trainer.load_from_checkpoint(ckpt_path, decoder=decoder_type, map_location=device)
model.eval()
net = model.getNet()

# load the valuation data to run inference on
data_root = Path("data/emp")
dataset = Av2Dataset(data_root=data_root, cached_split="val")
loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# take a single batch
batch = next(iter(loader))

# move to device (i use cpu because i dont have a GPU)
for k in batch:
    if torch.is_tensor(batch[k]):
        batch[k] = batch[k].to(device)

# calculate and output flops and the number of parameters in the model
flops, params = profile(net, inputs=(batch,))
print("Params (in Millions):", params / 1000000)
print("FLOPs (GFLOPs, 1 = billion operations):", flops / 1000000000)
