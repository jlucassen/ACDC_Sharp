from typing import Tuple, Optional, Dict, Union, Set
from transformer_lens import utils, HookedTransformer
from collections import defaultdict
from graphlib import TopologicalSorter

class TLGraph():
    graph: Dict[Tuple[str, Optional[int]], Set[Tuple[str, Optional[int]]]]

    def __init__(self, model: HookedTransformer) -> None:
        self.graph = defaultdict(set)
        

def topological_sort(graph: Dict[Tuple[str, Optional[int]], Set[Tuple[str, Optional[int]]]]):
    topo = TopologicalSorter(graph)
    return list(topo.static_order())
    
    
    
def get_node_key(hook_name: str, head_idx: Optional[int] = None) -> Tuple[str, Optional[int]]:
    if head_idx is None: 
        return (hook_name, None)
    else: 
        return (hook_name, head_idx)


def build_graph_from_model(model: HookedTransformer) -> Dict[Tuple[str, Optional[int]], 
                                                             Set[Tuple[str, Optional[int]]]]: 
    #keys are tuples that either is (layer_idx, head_idx) or (layer_idx, None)
    
    # setup
    graph = defaultdict(set)
    n_layers = model.cfg.n_layers
    downstream_resid_nodes = set() # downstream means later in the forward pass
    
    # start with the root node (last resid node)
    root_node = get_node_key(utils.get_act_name("resid_post", n_layers-1))
    downstream_resid_nodes.add(root_node)
    
    for layer_idx in range(n_layers - 1, -1, -1):

        # not supporting mlps right now 
        if not model.cfg.attn_only: 
            raise Exception("really need to implement this")
        new_downstream_resid_nodes = set()
        for head_idx in range(model.cfg.n_heads - 1, -1, -1):
            head_name = f"blocks.{layer_idx}.attn.hook_result"
            cur_node = get_node_key(head_name, head_idx)
            for resid_node in downstream_resid_nodes: 
                graph[cur_node].add(resid_node)
            
            # if model.cfg.use_split_qkv_input:
            for decomposed_name in ["q", "k", "v"]: 
                decomposed_node = get_node_key(
                                                utils.get_act_name(decomposed_name, layer_idx), 
                                                head_idx)
                decomposed_input_node = get_node_key(
                                                utils.get_act_name(f"{decomposed_name}_input", layer_idx), 
                                                head_idx)

                graph[decomposed_node].add(cur_node)
                graph[decomposed_input_node].add(decomposed_node)
                new_downstream_resid_nodes.add(decomposed_input_node)
        downstream_resid_nodes.update(new_downstream_resid_nodes)
                    
                
    # maybe implement no pos embed later
    
    tok_embed_node = get_node_key(utils.get_act_name("embed"))
    pos_embed_node = get_node_key(utils.get_act_name("pos_embed"))
    embed_nodes = [tok_embed_node, pos_embed_node]
    for embed_node in embed_nodes: 
        for resid_node in downstream_resid_nodes: 
            graph[embed_node].add(resid_node)

    return graph

if __name__ == "__main__": 
    pass 