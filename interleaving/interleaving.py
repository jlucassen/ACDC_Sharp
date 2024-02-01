# %%
import torch as t
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from functools import partial
from tqdm import tqdm

from EAP_utils import get_hooks_from_nodes

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
# %%
model = HookedTransformer.from_pretrained(
  'gpt2-small',
  center_writing_weights=False,
  center_unembed=False,
  fold_ln=False,
  device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
# %%
clean_dataset = IOIDataset(
  prompt_type='mixed',
  N=25,
  tokenizer=model.tokenizer,
  prepend_bos=False,
  seed=1,
  device=device
)
corrupt_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

# %%
def get_activations_with_grads(model, dataset):
 '''
 Takes:
 - a HookedTransformer model
 - a dataset
 Returns:
 - the logits from the forward pass
 - a dictionary of activations, keyed by hook name, with gradients
 '''

 # save activations by reference, so we can get grads. ActivationCache saves a copy.
 def save_reference_hook(activation, hook, save_dict):
  save_dict[hook.name] = activation
  return activation

 cache_with_grads = {} 

 logits = model.run_with_hooks(
    dataset.sentences,
    fwd_hooks = [(
     lambda s: True, # apply this hook everywhere
     partial(save_reference_hook, save_dict=cache_with_grads) # save references to cache_with_grads
    )]
 )

 return logits, cache_with_grads

clean_logits, clean_activations_with_grads = get_activations_with_grads(model, clean_dataset)
corrupt_logits, corrupt_activations_with_grads = get_activations_with_grads(model, corrupt_dataset)
# %%

def hook_filter(hook_point_name, valid_types = ['mlp', 'head']):
 return any([valid in hook_point_name for valid in valid_types])

# %%
