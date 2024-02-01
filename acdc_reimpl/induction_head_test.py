#%% 
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")


import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from TLGraph import *
import torch.nn.functional as F
from utils import kl_divergence, shuffle_tensor
from TLExperiment import TLExperiment


import huggingface_hub
#%% 
t.set_grad_enabled(False)

if t.cuda.is_available():
    device = t.device("cuda")
elif t.backends.mps.is_available():
    device = t.device("mps")
else:
    device = t.device("cpu")


MAIN = __name__ == "__main__"
#%% 
def get_model(device):
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",  # load Redwood's model
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
        device=device,
    )

    # standard ACDC options
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    return tl_model

def get_validation_data(num_examples=None, seq_len=None, device=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = t.load(validation_fname, map_location=device).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None, device=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = t.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]
    
#%% 
num_examples = 10
seq_len = 300
threshold = 0.8
data_seed=42

model = get_model(device=device)
validation_data_orig = get_validation_data(device=device)
mask_orig = get_mask_repeat_candidates(num_examples=None, device=device) # None so we get all
assert validation_data_orig.shape == mask_orig.shape
assert seq_len <= validation_data_orig.shape[1]-1
validation_slice = slice(0, num_examples)
validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
validation_mask = mask_orig[validation_slice, :seq_len].contiguous()

validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()
with t.no_grad():
    base_val_logprobs = F.log_softmax(model(validation_data), dim=-1).detach()
        
#%% 
metric = partial(
    kl_divergence,
    base_model_logprobs=base_val_logprobs,
    mask_repeat_candidates=validation_mask,
    last_seq_element_only=False,
    return_one_element=True
)
# save base_val_logprobs in a file 
#%% 
exp = TLExperiment(
    model=model,
    clean_ds=validation_data,
    corr_ds=validation_patch_data,
    metric=metric,
    threshold=threshold,
    device=device,
    debug=True
)
exp.graph.count_edges()

#%% 
for _ in tqdm(range(1)):
    exp.step()
exp.graph.count_edges()

#%% 
exp.graph.reverse_graph
#%% 
exp.graph.generate_reduced_reverse_graph()

# %%
for i in orig_exp.graph.reverse_graph.keys(): 
    print(i)
print(len(orig_exp.graph.reverse_graph.keys()))
# %%
exp.graph.graph

# %%
