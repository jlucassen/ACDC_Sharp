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
#%% 
t.set_grad_enabled(False)

if t.cuda.is_available():
    device = t.device("cuda")
elif t.backends.mps.is_available():
    device = t.device("mps")
else:
    device = t.device("cpu")


MAIN = __name__ == "__main__"


# # %%
# model = HookedTransformer.from_pretrained(
#     "gpt2-small",
#     center_unembed=True,
#     center_writing_weights=True,
#     fold_ln=True,
#     refactor_factored_attn_matrices=True,
#     device=device
# )

# #%% 

# prompt_format = [
#     "When John and Mary went to the shops,{} gave the bag to",
#     "When Tom and James went to the park,{} gave the ball to",
#     "When Dan and Sid went to the shops,{} gave an apple to",
#     "After Martin and Amy went to the park,{} gave a drink to",
# ]
# name_pairs = [
#     (" Mary", " John"),
#     (" Tom", " James"),
#     (" Dan", " Sid"),
#     (" Martin", " Amy"),
# ]

# # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
# prompts = [
#     prompt.format(name)
#     for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1]
# ]

# tokens = model.to_tokens(prompts, prepend_bos=True)
# # Move the tokens to the GPU
# tokens = tokens.to(device)
# # Run the model and cache all activations
# original_logits, cache = model.run_with_cache(tokens)

#%% 
# act_name = utils.get_act_name("resid_post", 0)
# cache[act_name].shape, cache.accumulated_resid().shape

#%% 
# for key in list(cache.keys()): 
#     print(key)
# %%
model = HookedTransformer.from_pretrained( 
                                          "attn-only-2l", 
                                            center_unembed=True,
                                            center_writing_weights=True,
                                            fold_ln=True,
                                            refactor_factored_attn_matrices=True,
                                            device=device)
#%% 
print("model.cfg.use_split_qkv_input", model.cfg.use_split_qkv_input)
tlgraph = TLGraph(model)
# for key, value in tlgraph.reverse_graph.items():
#     print(key, value)
print(tlgraph.topo_order)
#%% 
