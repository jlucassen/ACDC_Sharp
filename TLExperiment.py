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
        threshold, 
        debug=False,
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
        self.online_cache = OrderedDict() 
        self.corrupted_cache = OrderedDict()
        
        self.sender_hook_dict = defaultdict(set)
        self.receiver_hook_dict = defaultdict(set)
        self.debug = debug
    
    def set_corrupted_cache(self):
        self.model.reset_hooks()
        self.model.cache_all(self.corrupted_cache)
        self.model(self.corr_ds)
        self.model.reset_hooks()
        
        
    def sender_hook(self, activations: Tensor, 
                          hook: HookPoint,
                          node: TLNodeIndex):
        activations_dup = activations
        if device is not None:
            activations_dup = activations_dup.to(device)
        if node.index is not None:
            activations_dup = activations_dup[node.torchlike_index()]
        idx = node.torchlike_index()
        self.online_cache[node.name][idx] = activations_dup[idx]
        return z
    
    def receiver_hook(self, activations: Tensor, 
                            hook: HookPoint,
                            node: TLNodeIndex):
        cur_incoming_edge_type = get_incoming_edge_type(node)
        
        if cur_incoming_edge_type == EdgeType.DIRECT_COMPUTATION: 
            idx = node.torchlike_index()
            if not self.reverse_graph[node]:
                parent_idx = parent_node.torchlike_index()
                activations[:][idx] = self.corrupted_cache[node.name][idx].to(activations.device)
            return activations
        elif cur_incoming_edge_type == EdgeType.ADDITION:
            cur_idx = node.torchlike_index()
            activations[:][cur_idx] = self.corrupted_cache[node.name][cur_idx].to(activations.device)
            
            for parent_node in self.graph.reverse_graph[node]:
                parent_idx = parent_node.torchlike_index()
                activations[cur_idx] += self.online_cache[parent_node.name][parent_idx].to(activations.device)
                activations[cur_idx] -= self.corrupted_cache[node.name][cur_idx].to(activations.device)
            return activations
        else:
            return activations
        
    def add_sender_hook(self, node: TLNodeIndex):
        
        if node.name in self.sender_hook_dict and \
            (node.index is None or node.index in self.sender_hook_dict[node.name]):
            raise Exception(f"sender hook already exists {node}")
        
        self.model.add_hook(
            name=node.name, 
            hook=partial(self.sender_hook, node=node),
        )
        self.sender_hook_dict[node.name].add(node.index)
    
    def add_receiver_hook(self, node: TLNodeIndex):
        if node.name in self.receiver_hook_dict and \
            (node.index is None or node.index in self.receiver_hook_dict[node.name]):
            raise Exception(f"receiver hook already exists {node}")
        
        self.model.add_hook(
            name=node.name, 
            hook=partial(self.receiver_hook, node=node)
        )
        self.receiver_hook_dict[node.name].add(node.index)
    
    def run_model_and_eval(self):
        logits = self.model(self.clean_ds)
        return self.metric(logits)

    def try_remove_edges(self, cur_node):
        cur_incoming_edge_type = get_incoming_edge_type(cur_node)
        
        for parent_node in self.graph.reverse_graph[cur_node]:
            if cur_incoming_edge_type == EdgeType.ADDITION:
                self.add_sender_hook(parent_node)
                
            self.graph.remove_edge(parent_node, cur_node)
            old_eval = self.cur_eval
            new_eval = self.run_model_and_eval()
            
            if new_eval - old_eval < self.threshold:
                print(f"removing edge {parent_node} -> {cur_node}")
                self.cur_eval = new_eval
            else: 
                self.graph.add_edge(parent_node, cur_node)
            
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
            self.add_sender_hook(cur_node)
        
        if cur_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"] \
            or cur_incoming_edge_type == TLEdgeType.PLACEHOLDER:
            pass
        else: 
            self.try_remove_edges(cur_node)
            
        self.current_node_idx += 1
        
            
        

        


