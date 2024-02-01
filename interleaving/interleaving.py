# %%
import torch as t
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import einops

from acdc_reimpl.TLGraph import TLGraph, TLNodeIndex

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

# calculate activation diff

# %%

threshold = 0

metric_value.backward()

graph = TLGraph(model)
graph.reverse_graph[TLNodeIndex('blocks.11.hook_resid_post')]

included_grads_to_metric = defaultdict([]) # lists of grads to metric along included path, keyed by receiver name
included_grads_to_metric[graph.topo_order[0]] = list(t.autograd.grad(outputs=metric_value, inputs=clean_logits),
                                                    t.autograd.grad(outputs=metric_value, inputs=corrupt_logits))

for receiver in graph.topo_order:
 if receiver in included_grads_to_metric.keys(): # check if the recieving node has any included grads to metric
  grads_to_metric = included_grads_to_metric[receiver]
 else: # if not, ignore this receiver
  continue
 for sender in graph.reverse_graph[receiver]:
  clean_sender_activation = clean_activations_with_grads[sender.name][sender.torchlike_index()]
  corrupt_sender_activation = corrupt_activations_with_grads[sender.name][sender.torchlike_index()]
  sender_activation_diff = clean_sender_activation - corrupt_sender_activation

  clean_receiver_activation = clean_activations_with_grads[receiver.name][sender.torchlike_index()]
  corrupt_receiver_activation = corrupt_activations_with_grads[receiver.name][sender.torchlike_index()]
  clean_grad = t.autograd.grad(outputs=clean_receiver_activation, inputs=clean_sender_activation, retain_graph=True)
  corrupt_grad = t.autograd.grad(outputs=corrupt_receiver_activation, inputs=corrupt_sender_activation, retain_graph=True)
  
  clean_reciever_activation_diff = einops.einsum(sender_activation_diff, clean_grad, 'a b, b c -> a c')
  corrupt_reciever_activation_diff = einops.einsum(sender_activation_diff, corrupt_grad, 'a b, b c -> a c')

  for grad_to_metric in grads_to_metric:
    clean_metric_diff = t.dot(clean_reciever_activation_diff, grad_to_metric) # sender diff is clean-corrupt, so this gives metric improvement
    corrupt_metric_diff = t.dot(corrupt_reciever_activation_diff, grad_to_metric)

    if clean_metric_diff > threshold:
     included_grads_to_metric[sender].append(clean_grad @ grad_to_metric)
    elif corrupt_metric_diff > threshold:
     included_grads_to_metric[sender].append(corrupt_grad @ grad_to_metric)
     


# %%
