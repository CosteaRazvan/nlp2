import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer

from src.models import Classificator 
from src.utils import get_dataloader
from src.config import ModelConfig
from src.train import train 
from src.config import config

from verisnap import make_snapshot


seed = 314 

import numpy as np
np.random.seed(seed)
np.random.RandomState(seed)

import random
random.seed(seed)

import torch
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)


# Set environment variable for deterministic algorithms
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Encoder name
# encoder_name = "bert-base-multilingual-cased"
# encoder_name = "xlm-roberta-base"
# encoder_name = "readerbench/RoBERT-base"
# encoder_name = "BAAI/bge-m3"


path = '/mnt/storage/Code/nlp/data/train_data.csv'

for encoder_id in ["bert-base-multilingual-cased", "xlm-roberta-base", "readerbench/RoBERT-base"]:

    for encoder_freeze in [True, False]:

        encoder_name = encoder_id + f"-{'encoder_freeze' if encoder_freeze else 'encoder_trained'}"

        source_dir = '/mnt/storage/Code/nlp/src/'  # Directory containing the experiment results
        snapshots_dir = f'/mnt/storage/Code/nlp/snpashots/{encoder_name}'  # Directory where snapshots will be stored
        threshold = 50  # MB, files larger than this will be symbolically linked

        version = make_snapshot(source_dir, snapshots_dir, threshold)


        config.name_model = encoder_name.replace("/", "_")

        # Initialize model
        model = Classificator(encoder_id, encoder_freeze)
        model.to(config.device)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(encoder_id)

        # Get data loaders
        train_loader, val_loader = get_dataloader(config, tokenizer, path)

        # Start training
        train(config, model, train_loader, val_loader, version)