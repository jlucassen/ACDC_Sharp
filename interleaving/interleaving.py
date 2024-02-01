# %%
import torch as t
from torch import Tensor
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import einops

from TLInterleavingGraph import TLGraph, TLNodeIndex
from jaxtyping import Float

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
    dataset.toks,
    fwd_hooks = [(
     lambda s: True, # apply this hook everywhere
     partial(save_reference_hook, save_dict=cache_with_grads) # save references to cache_with_grads
    )]
 )

 return logits, cache_with_grads

clean_logits, clean_cache = get_activations_with_grads(model, clean_dataset)
corrupt_logits, corrupt_cache = get_activations_with_grads(model, corrupt_dataset)
# calculate activation diff
def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    batch_size = logits.size(0)
    io_logits = logits[range(batch_size), ioi_dataset.word_idx['end'][:batch_size], ioi_dataset.io_tokenIDs[:batch_size]]
    s_logits = logits[range(batch_size), ioi_dataset.word_idx['end'][:batch_size], ioi_dataset.s_tokenIDs[:batch_size]]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()
  
clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset)
corrupt_logit_diff = ave_logit_diff(corrupt_logits, corrupt_dataset)

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)
    
# Get clean and corrupt logit differences
clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corrupt_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')

#%% 
# clean_logit_diff.backward()
# corrupt_logit_diff.backward()
# %%
# threshold = 0
graph = TLGraph(model)

end_node = graph.topo_order[0]
end_node_clean_act = clean_cache[end_node.name]
end_node_corrupt_act = corrupt_cache[end_node.name]

print(end_node_clean_act.requires_grad, clean_logit_diff.requires_grad)
clean_grad = t.autograd.grad(outputs=clean_logit_diff, inputs=end_node_clean_act, retain_graph=True)[0]
corrupt_grad = t.autograd.grad(outputs=corrupt_logit_diff, inputs=end_node_corrupt_act, retain_graph=True)[0]


#%% 
print(clean_grad.shape)

# need to initialize the grad from the end node to the metric, so we can start path finding
end_node_clean_corrupt = clean_cache[end_node.name] - corrupt_cache[end_node.name]
end_node_corrupt_clean = corrupt_cache[end_node.name] - clean_cache[end_node.name]
print(end_node_corrupt_clean.shape, clean_grad.shape)
end_node.grad = t.max(einops.einsum(end_node_corrupt_clean, clean_grad, 'a b c, a b c ->'), einops.einsum(end_node_clean_corrupt, corrupt_grad, 'a b c, a b c ->'))
print(end_node.grad)
#%% 
for receiver in graph.topo_order:
    children = graph.reverse_graph[receiver]
    for sender in children:
        # edge = graph.edges[sender][receiver]
        # if edge.type == TLEdgeType.PLACEHOLDER:
        #     continue
        # if edge.visited: 
        #     continue
        # edge.visited = True
        
        clean_sender_act = clean_cache[sender.name][sender.torchlike_index()]
        corrupt_sender_act = corrupt_cache[sender.name][sender.torchlike_index()]
        
        clean_receiver_act = clean_cache[receiver.name][receiver.torchlike_index()]
        corrupt_receiver_act = corrupt_cache[receiver.name][receiver.torchlike_index()]
        
        clean_grad = t.autograd.grad(outputs=clean_receiver_act, inputs=clean_sender_act, retain_graph=True)
        corrupt_grad = t.autograd.grad(outputs=corrupt_receiver_act, inputs=corrupt_sender_act, retain_graph=True)
        
        clean_corrupt_diff = clean_sender_act - corrupt_sender_act
        corrupt_clean_diff = corrupt_sender_act - clean_sender_act
        
        running_corrupt_grad_score = clean_corrupt_diff @ corrupt_grad @ receiver.grad
        running_clean_grad_score = corrupt_clean_diff @ clean_grad @ receiver.grad
        
        if sender.grad is None: 
          if running_corrupt_grad_score > running_clean_grad_score:
            sender.grad = corrupt_grad
            sender.score = running_corrupt_grad_score
          else: 
            sender.grad = -clean_grad
            sender.score = running_clean_grad_score
        else: 
          if running_corrupt_grad_score > sender.score:
            sender.grad = corrupt_grad
            sender.score = running_corrupt_grad_score
          elif running_clean_grad_score > sender.score:
            sender.grad = -clean_grad
            sender.score = running_clean_grad_score
          
#%% 
        


# #%% 

#  if receiver in included_grads_to_metric.keys(): # check if the recieving node has any included grads to metric
#   grads_to_metric = included_grads_to_metric[receiver]
#  else: # if not, ignore this receiver
#   continue
#  for sender in graph.reverse_graph[receiver]:
  

#   clean_receiver_activation = clean_cache[receiver.name][sender.torchlike_index()]
#   corrupt_receiver_activation = corrupt_activations_with_grads[receiver.name][sender.torchlike_index()]
#   clean_grad = t.autograd.grad(outputs=clean_receiver_activation, inputs=clean_sender_activation, retain_graph=True)
#   corrupt_grad = t.autograd.grad(outputs=corrupt_receiver_activation, inputs=corrupt_sender_activation, retain_graph=True)
  
#   clean_reciever_activation_diff = einops.einsum(sender_activation_diff, clean_grad, 'a b, b c -> a c')
#   corrupt_reciever_activation_diff = einops.einsum(sender_activation_diff, corrupt_grad, 'a b, b c -> a c')

#   for grad_to_metric in grads_to_metric:
#     clean_metric_diff = t.dot(clean_reciever_activation_diff, grad_to_metric) # sender diff is clean-corrupt, so this gives metric improvement
#     corrupt_metric_diff = t.dot(corrupt_reciever_activation_diff, grad_to_metric)

#     if clean_metric_diff > threshold:
#      included_grads_to_metric[sender].append(clean_grad @ grad_to_metric)
#     elif corrupt_metric_diff > threshold:
#      included_grads_to_metric[sender].append(corrupt_grad @ grad_to_metric)
     


# %%
