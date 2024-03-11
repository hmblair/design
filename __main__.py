# __main__.py

import os 
import torch
from tqdm.auto import tqdm
from pytorch_lightning.cli import LightningCLI

# create the logs/wandb directory if it does not exist
if not os.path.exists('logs/wandb'):
    os.makedirs('logs/wandb')

# set the WANDB_PROJECT environment variable to the name of the current directory
os.environ['WANDB_PROJECT'] = os.path.basename(os.getcwd())

# set the default precision for matmul operations to medium
torch.set_float32_matmul_precision('medium')

# clear the tqdm instances, to prevent bugs with the progress bar
tqdm._instances.clear()

# run the LightningCLI
if __name__ == "__main__":
    LightningCLI(
        save_config_kwargs={"overwrite": True, 'multifile' : True},
        )

"""
Example usage for fit, test, and predict:

python3 __main__.py fit \
    --config examples/phase/fit.yaml \
    --config examples/model/lstm.yaml
    
python3 __main__.py test \
    --config examples/phase/test.yaml \
    --config examples/model/lstm.yaml \
    --ckpt_path models/checkpoints/lstm.ckpt

python3 __main__.py predict \
    --config examples/phase/predict.yaml \
    --config examples/model/lstm.yaml \
    --ckpt_path models/checkpoints/lstm.ckpt
"""