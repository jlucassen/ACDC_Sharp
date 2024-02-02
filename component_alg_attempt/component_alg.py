# %%
import torch as t
from transformer_lens import HookedTransformer
from ioi_dataset import IOIDataset
from functools import partial

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
def get_attn_hooks_and_heads(model):
    n_heads = model.cfg.n_heads
    attn_hooks = [name for name in model.hook_dict.keys() if name.endswith('_z')]
    attn_heads = [[name+f"[{num}]" for num in range(n_heads)] for name in attn_hooks]
    attn_heads_flat = sum(attn_heads, [])
    return attn_hooks, attn_heads_flat

def my_act_saver_hook(activation, hook, cache, n_heads):
    if hook.endswith('_z'): # for attn heads
        for head in range(n_heads):
            cache[hook+f"[{head}]"] = activation[:, :, head, :]
    else: # for final resid stream
        cache[hook] = activation.clone()
    return activation

def calculate_activation_diffs(clean_cache, corrupt_cache):
    diffs = {}
    for key in clean_cache.keys():
        assert key in corrupt_cache
        diffs[key] = clean_cache[key] - corrupt_cache[key]
    return diffs

def my_patcher_hook(activation, hook, cache, head):
    if hook.endswith('_z'): # for attn heads
        activation[:, :, head, :] = cache[hook+f"[{head}]"]
    else: # should never be patching anything besides attn heads
        raise NotImplementedError 
    return activation

def mean_dots(first, second):
    dots = []
    for key in first.keys():
        assert key in second
        dots.append(t.dot(first[key], second[key]))
    return t.mean(dots)
          
# %%
def main(model, clean_ds, corrupt_ds, threshold, n_iter):
    # get attn head names
    attn_hooks, attn_heads = get_attn_hooks_and_heads(model)
    # get final resid name
    final_resid_name = f"blocks.{model.cfg.n_heads-1}.hook_resid_post"
    # those are all the activations we read
    acts_to_read = attn_hooks+[final_resid_name]
    # get clean activations
    clean_cache = {}
    act_saver_hook_curried = partial(my_act_saver_hook, cache = clean_cache, n_heads = model.cfg.n_heads)
    clean_logits = model.run_with_hooks(clean_ds, fwd_hooks=[(acts_to_read, act_saver_hook_curried)])
    model.reset_hooks()
    # get corrupted activations
    corrupt_cache = {}
    act_saver_hook_curried = partial(my_act_saver_hook, cache = corrupt_cache, n_heads = model.cfg.n_heads)
    corrupt_logits = model.run_with_hooks(corrupt_ds, fwd_hooks=[(acts_to_read, act_saver_hook_curried)])
    model.reset_hooks()
    # get activation diffs at all heads
    og_act_diffs = calculate_activation_diffs(clean_cache, corrupt_cache)
    
    # start with final resid stream included
    included_heads = [final_resid_name]
    for i in n_iter:
        do_noise = bool(i%2) # alternate noising and denoising, start with noising
        if do_noise:
            baseline_cache = clean_cache
            baseline_ds = clean_ds
            patch_cache = corrupt_cache
        else:
            baseline_cache = corrupt_cache
            baseline_ds = corrupt_ds
            patch_cache = clean_cache
        attn_heads_to_include = []
        for attn_hook in attn_hooks:
            for head in range(model.cfg.n_heads):
                # patch the head, get patched activations
                my_patcher_hook_curried = partial(my_patcher_hook, cache = patch_cache, head = head)
                patched_act_cache = {}
                act_saver_hook_curried = partial(my_act_saver_hook, cache = patched_act_cache, n_heads = model.cfg.n_heads)
                model.run_with_hooks(
                    baseline_ds,
                    fwd_hooks = [
                        (attn_hook, my_patcher_hook_curried), # my_patcher_hook_curried has head index curried in
                        (included_heads, act_saver_hook_curried)
                    ]
                )
                model.reset_hooks()
                # get mean of act diffs at included heads
                patched_act_diffs = calculate_activation_diffs(patched_act_cache, baseline_cache)
                score = mean_dots(patched_act_diffs, og_act_diffs)
                # decide whether or not to include head as a new sensor
                print(attn_hook+f"[{head}]", score)
                if score > threshold:
                    attn_heads_to_include.append(attn_hook+f"[{head}]")
        included_heads += attn_heads_to_include
    return included_heads
# %%
