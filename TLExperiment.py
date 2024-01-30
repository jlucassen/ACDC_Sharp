from transformer_lens.HookedTransformer import HookedTransformer

from graph_builder import TLGraph, get_incoming_edge_type
from collections import OrderedDict

from torch import Tensor
class TLExperiment:

    def __init__(
        self,
        model: HookedTransformer,
        clean_ds,
        corr_ds,
        metric,
        threshold
    ):
        self.model = model
        self.clean_ds = clean_ds
        self.corr_ds = corr_ds
        self.metric = metric
        self.threshold = threshold

        self.model.reset_hooks()
        self.graph = TLGraph(model)
        self.current_node_idx = 0
        self.steps = 0
        self.cache = OrderedDict() 
    
    # def patch_hook(self, hook_point):
    #     activations_to_patch = self.corr_cache[hook_point]
    #     self.model.hook_dict[hook_point][:] = activations_to_patch
    #     return self.model.hook_dict[hook_point]
        
    # def freeze_hook(self, hook_point):
    #     activations_to_patch = self.corr_cache[hook_point]
    #     self.model.hook_dict[hook_point][:] = activations_to_patch
    #     return self.model.hook_dict[hook_point]
    
    
    def sender_hook(self, activations: Tensor, 
                          hook: HookPoint,
                          node: TLNodeIndex):
        activations_dup = activations
        if device is not None:
            activations_dup = activations_dup.to(device)
        if node.index is not None:
            activations_dup = activations_dup[node.torchlike_index()]
            
        self.cache[node] = activations_dup
        return z
    
    def receiver_hook(self, activations: Tensor, 
                            hook: HookPoint,
                            node: TLNodeIndex):
        cur_incoming_edge_type = get_incoming_edge_type(node)
        
        if cur_incoming_edge_type == EdgeType.DIRECT_COMPUTATION: 
            idx = node.torchlike_index()
            activations[:][idx] = self.corrupted_cache[node.name][idx].to(activations.device)
            return activations
        elif cur_incoming_edge_type == EdgeType.ADDITION:
            idx = node.torchlike_index()
            activations[:][idx] = self.corrupted_cache[node.name][idx].to(activations.device)
            return activations
        else:
            return activations
    
    
    def run_model_and_eval(self):
        logits = self.model(self.clean_ds)
        return self.metric(logits)

    def try_remove_edges(self, cur_node):
        cur_incoming_edge_type = get_incoming_edge_type(cur_node)
        
        for parent_node in self.graph.reverse_graph[cur_node]:
            if cur_incoming_edge_type == EdgeType.ADDITION:
                added_sender_hook = self.add_sender_hook(parent_node)
            else:
                added_sender_hook = False
                
            old_eval = self.cur_eval
            new_eval = self.run_model_and_eval()
            
            if new_eval - old_eval < self.threshold:
                self.cur_eval = new_eval
                self.graph.remove_edge(parent_node, cur_node)    
            
    def step(self):
        if current_node_idx < 0:
            return 
        cur_node = self.graph.reverse_topo_order[self.current_node_idx]
        self.steps += 1
        
        self.cur_eval = self.run_model_and_eval()
        
        cur_incoming_edge_type = get_incoming_edge_type(current_node)
        if cur_incoming_edge_type != TLEdgeType.PLACEHOLDER:
            self.add_receiver_hook(cur_node)
            
        if cur_incoming_edge_type == TLEdgeType.DIRECT_COMPUTATION:
            self.add_sender_hook(cur_node, override=True)
        
        if cur_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"] \
            or cur_incoming_edge_type == TLEdgeType.PLACEHOLDER:
            pass
        else: 
            self.try_remove_edges(cur_node)
            
        self.current_node_idx += 1
        
            
        

        


