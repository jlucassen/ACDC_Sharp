# from typing import Tuple, Optional, Dict, Union, Set
# from transformer_lens import utils, HookedTransformer
# from collections import defaultdict
# from graphlib import TopologicalSorter
# from enum import Enum

# class TLNodeIndex:
#     name: str 
#     index: Optional[int]
#     def __init__(self, name: str, index: Optional[int] = None):
#         self.name = name
#         self.index = index
#         self.grad = None
#         self.score = None
    
#     def __eq__(self, other):
#         assert isinstance(other, TLNodeIndex)
#         return self.name == other.name and self.index == other.index
    
#     def __repr__(self):
#         if self.index is None:
#             return self.name
#         return f"{self.name}[{self.index}] score:{self.score}"
    
#     def __hash__(self):
#         return hash(self.__repr__())    
    
#     def __gt__(self, other):
#         assert isinstance(other, TLNodeIndex)
#         if self.name == other.name: 
#             return self.index > other.index
#         else:
#             return self.name < other.name
    
#     def torchlike_index(self):
#         if self.index is None:
#             return slice(None)
#         else: 
#             return tuple([slice(None)] * 2 + [self.index])
    
    
# class TLEdgeType(Enum):
#     ADDITION = 0
#     DIRECT_COMPUTATION = 1
#     PLACEHOLDER = 2
    
#     def __eq__(self, other):
#         """Necessary because of extremely frustrating error that arises with load_ext autoreload (because this uses importlib under the hood: https://stackoverflow.com/questions/66458864/enum-comparison-become-false-after-reloading-module)"""

#         if type(self).__qualname__ != type(other).__qualname__:
#             return NotImplemented
#         return self.name == other.name and self.value == other.value
    
    
# # class TLEdge:
# #     type: TLEdgeType
# #     use_clean_grad: bool
# #     visited: bool
# #     present: bool 
    
# #     def __init__(self, type: TLEdgeType):
# #         self.type = type
# #         self.use_clean_grad = None
# #         self.present = True
# #         self.clean_grad = None 
# #         self.corrupt_grad = None
         
    

# def get_incoming_edge_type(child_node: TLNodeIndex) -> TLEdgeType:
#     # parent_layer, parent_head = parent_node
#     child_layer = child_node.name
    
#     if child_layer.endswith("result") or child_layer.endswith("z") or child_layer.endswith("mlp_out"):
#         return TLEdgeType.PLACEHOLDER
#     elif child_layer.endswith("resid_post") or child_layer.endswith("_input"):
#         return TLEdgeType.ADDITION
#     else:
#         return TLEdgeType.DIRECT_COMPUTATION
    
    
# class TLGraph():
    
#     graph: Dict[TLNodeIndex, Set[TLNodeIndex]]

#     def __init__(self, model: HookedTransformer, use_pos_embed=False) -> None:
#         self.graph = defaultdict(set)
#         self.reverse_graph = defaultdict(set)
#         self.cfg = model.cfg
#         self.use_pos_embed = use_pos_embed
#         self.build_graph()
#         self.build_reverse_graph()
#         # self.build_edges()
#         self.topological_sort()
#         self.reverse_topo_order = self.topo_order[::-1]
        
        
#     def build_graph(self) -> None:
#         n_layers = self.cfg.n_layers
#         downstream_resid_nodes = set() # downstream means later in the forward pass
    
#         # start with the root node (last resid node)
#         root_node = TLNodeIndex(utils.get_act_name("resid_post", n_layers-1))
#         downstream_resid_nodes.add(root_node)
        
#         for layer_idx in range(n_layers - 1, -1, -1):

#             # not supporting mlps right now 
#             if not self.cfg.attn_only: 
#                 mlp_out_node = TLNodeIndex(f"blocks.{layer_idx}.hook_mlp_out")
#                 mlp_in_node = TLNodeIndex(f"blocks.{layer_idx}.hook_mlp_in")
#                 for resid_node in downstream_resid_nodes: 
#                     self.graph[mlp_out_node].add(resid_node)
#                 self.graph[mlp_in_node].add(mlp_out_node)
#                 downstream_resid_nodes.add(mlp_in_node)
                
                
                
#             new_downstream_resid_nodes = set()
#             for head_idx in range(self.cfg.n_heads - 1, -1, -1):
#                 if self.cfg.use_attn_result: 
#                     head_name = f"blocks.{layer_idx}.attn.hook_result"
#                 else: 
#                     head_name = utils.get_act_name("z", layer_idx)
#                 cur_node = TLNodeIndex(head_name, head_idx)
#                 for resid_node in downstream_resid_nodes: 
#                     self.graph[cur_node].add(resid_node)
                
#                 # if model.cfg.use_split_qkv_input:
#                 for decomposed_name in ["q", "k", "v"]: 
#                     decomposed_node = TLNodeIndex(
#                                                     utils.get_act_name(decomposed_name, layer_idx), 
#                                                     head_idx)
#                     decomposed_input_node = TLNodeIndex(
#                                                     utils.get_act_name(f"{decomposed_name}_input", layer_idx), 
#                                                     head_idx)

#                     self.graph[decomposed_node].add(cur_node)
#                     self.graph[decomposed_input_node].add(decomposed_node)
#                     new_downstream_resid_nodes.add(decomposed_input_node)
#             downstream_resid_nodes.update(new_downstream_resid_nodes)
                        
#         if self.use_pos_embed:
#             tok_embed_node = TLNodeIndex(utils.get_act_name("embed"))
#             pos_embed_node = TLNodeIndex(utils.get_act_name("pos_embed"))
#             embed_nodes = [tok_embed_node, pos_embed_node]
#             for embed_node in embed_nodes: 
#                 for resid_node in downstream_resid_nodes: 
#                     self.graph[embed_node].add(resid_node)
#         else: 
#             resid_pre = TLNodeIndex(utils.get_act_name("resid_pre", 0))
#             for resid_node in downstream_resid_nodes: 
#                 self.graph[resid_pre].add(resid_node)
            
                
#     def topological_sort(self):
#         topo = TopologicalSorter(self.graph)
#         self.topo_order = list(topo.static_order())
#         return self.topo_order
    
#     def build_reverse_graph(self):
#         for node in self.graph: 
#             for child in self.graph[node]: 
#                 self.reverse_graph[child].add(node)
                
                
#     # def build_edges(self):
#     #     self.edges = defaultdict(defaultdict(TLEdge))
#     #     for parent in self.graph:
#     #         for child in self.graph[parent]:
#     #             edge_type = get_incoming_edge_type(child)
#     #             self.edges[parent][child] = TLEdge(edge_type, True)
        
        
#     def __getitem__(self, node: TLNodeIndex):
#         return self.graph[node]
    
#     def add_edge(self, sender: TLNodeIndex, receiver: TLNodeIndex):
#         self.graph[sender].add(receiver)
#         self.reverse_graph[receiver].add(sender)
    
#     def remove_edge(self, sender: TLNodeIndex, receiver: TLNodeIndex):
#         self.graph[sender].remove(receiver)
#         self.reverse_graph[receiver].remove(sender)
    
#     def count_edges(self): 
#         return sum([len(self.reverse_graph[node]) \
#                     if get_incoming_edge_type(node) != TLEdgeType.PLACEHOLDER \
#                     else 0 \
#                     for node in self.reverse_graph])
    
#     def node_disconnected(self, node: TLNodeIndex):
#         return len(self.graph[node]) == 0
    
#     def remove_node(self, node: TLNodeIndex):
#         all_parents = self.reverse_graph[node].copy() 
#         for parent in all_parents:
#             self.remove_edge(parent, node)
            
#     def generate_reduced_reverse_graph(self):
#         reduced_graph = {}
#         for node in self.reverse_graph: 
#             if self.reverse_graph[node]:
#                 reduced_graph[node] = self.reverse_graph[node]
            
#         return reduced_graph
            
    
# if __name__ == "__main__":    
#     node_index = TLNodeIndex("test", 1)
#     print(node_index.torchlike_index())
#     node_index = TLNodeIndex("test")
#     print(node_index.torchlike_index())