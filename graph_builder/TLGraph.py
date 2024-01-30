from typing import Tuple, Optional, Dict, Union, Set
from transformer_lens import utils, HookedTransformer
from collections import defaultdict
from graphlib import TopologicalSorter
from enum import Enum

class TLNodeIndex:
    name: str 
    index: Optional[int]
    def __init__(self, name: str, index: Optional[int] = None):
        self.name = name
        self.index = index
    
    def __eq__(self, other):
        assert isinstance(other, NodeIndex)
        return self.name == other.name and self.index == other.index
    
    def __repr__(self):
        if self.index is None:
            return self.name
        return f"{self.name}[{self.index}]"
    
    def __hash__(self):
        return hash(self.__repr__())    
    
    
class TLEdgeType(Enum):
    ADDITION = 0
    DIRECT_COMPUTATION = 1
    PLACEHOLDER = 2
    
    def __eq__(self, other):
        """Necessary because of extremely frustrating error that arises with load_ext autoreload (because this uses importlib under the hood: https://stackoverflow.com/questions/66458864/enum-comparison-become-false-after-reloading-module)"""

        assert isinstance(other, EdgeType)
        return self.value == other.value
    

def get_incoming_edge_type(child_node: TLNodeIndex) -> TLEdgeType:
    # parent_layer, parent_head = parent_node
    child_layer, child_head = child_node
    
    if child_layer.endswith("attn_result") or child_layer.endswith("z"):
        return TLEdgeType.PLACEHOLDER
    elif child_layer.endswith("resid_post") or child_layer.endswith("_input"):
        return TLEdgeType.ADDITION
    else:
        return TLEdgeType.DIRECT_COMPUTATION
    
    
class TLGraph():
    
    graph: Dict[TLNodeIndex, Set[TLNodeIndex]]

    def __init__(self, model: HookedTransformer) -> None:
        self.graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.model = model
        self.cfg = model.cfg
        self.build_graph()
        self.build_reverse_graph()
        self.topological_sort()
        self.reverse_topo_order = self.topo_order[::-1]
        
        
    def build_graph(self) -> None:
        n_layers = self.cfg.n_layers
        downstream_resid_nodes = set() # downstream means later in the forward pass
    
        # start with the root node (last resid node)
        root_node = TLNodeIndex(utils.get_act_name("resid_post", n_layers-1))
        downstream_resid_nodes.add(root_node)
        
        for layer_idx in range(n_layers - 1, -1, -1):

            # not supporting mlps right now 
            if not self.cfg.attn_only: 
                raise Exception("really need to implement this")
            new_downstream_resid_nodes = set()
            for head_idx in range(self.cfg.n_heads - 1, -1, -1):
                if self.cfg.use_attn_result: 
                    head_name = f"blocks.{layer_idx}.attn.attn_result"
                else: 
                    head_name = utils.get_act_name("z", layer_idx)
                cur_node = TLNodeIndex(head_name, head_idx)
                for resid_node in downstream_resid_nodes: 
                    self.graph[cur_node].add(resid_node)
                
                # if model.cfg.use_split_qkv_input:
                for decomposed_name in ["q", "k", "v"]: 
                    decomposed_node = TLNodeIndex(
                                                    utils.get_act_name(decomposed_name, layer_idx), 
                                                    head_idx)
                    decomposed_input_node = TLNodeIndex(
                                                    utils.get_act_name(f"{decomposed_name}_input", layer_idx), 
                                                    head_idx)

                    self.graph[decomposed_node].add(cur_node)
                    self.graph[decomposed_input_node].add(decomposed_node)
                    new_downstream_resid_nodes.add(decomposed_input_node)
            downstream_resid_nodes.update(new_downstream_resid_nodes)
                        
                    
        # maybe implement no pos embed later
        
        tok_embed_node = TLNodeIndex(utils.get_act_name("embed"))
        pos_embed_node = TLNodeIndex(utils.get_act_name("pos_embed"))
        embed_nodes = [tok_embed_node, pos_embed_node]
        for embed_node in embed_nodes: 
            for resid_node in downstream_resid_nodes: 
                self.graph[embed_node].add(resid_node)
                
    def topological_sort(self):
        topo = TopologicalSorter(self.graph)
        self.topo_order = list(topo.static_order())
        return self.topo_order
    
    def build_reverse_graph(self):
        for node in self.graph: 
            for child in self.graph[node]: 
                self.reverse_graph[child].add(node)
        
    def __getitem__(self, node: TLNodeIndex):
        return self.graph[node]
    
                
    
    
    
    
if __name__ == "__main__": 
    pass 
        