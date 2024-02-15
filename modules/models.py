from modules import shared
import torch
import gc

def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        torch.cuda.empty_cache()