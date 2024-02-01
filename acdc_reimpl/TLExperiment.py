from transformer_lens.HookedTransformer import HookedTransformer, HookPoint
from TLGraph import TLGraph, get_incoming_edge_type, TLNodeIndex, TLEdgeType
from collections import OrderedDict, defaultdict
from functools import partial 
from torch import Tensor
import torch as t
class TLExperiment:

    def __init__(
        self,
        model: HookedTransformer,
        clean_ds,
        corr_ds,
        metric,
        threshold, 
        device,
        debug=False,
    ):
        self.model = model
        self.clean_ds = clean_ds
        self.corr_ds = corr_ds
        self.metric = lambda x: metric(x).item()
        self.threshold = threshold

        self.model.reset_hooks()
        self.graph = TLGraph(model)
        self.current_node_idx = 0
        self.steps = 0
        self.online_cache = OrderedDict() 
        self.corrupted_cache = OrderedDict()
        self.set_corrupted_cache()
        self.sender_hook_dict = defaultdict(set)
        self.receiver_hook_dict = defaultdict(set)
        self.device = device
        self.debug = debug
        
        
        self.add_all_sender_hooks()
        
        if self.debug: 
            print(f"edges: {self.graph.count_edges()}")
            print(f"corrupted cache keys: {self.corrupted_cache.keys()}")
            
    
    def set_corrupted_cache(self):
        self.model.reset_hooks()
        self.model.cache_all(self.corrupted_cache)
        self.model(self.corr_ds)
        self.model.reset_hooks()
        
        
    def sender_hook(self, activations: Tensor, 
                          hook: HookPoint):
        activations_dup = activations
        if self.device is not None:
            activations_dup = activations_dup.to(self.device)
        # if node.index is not None:
        #     activations_dup = activations_dup[node.torchlike_index()]
        # idx = node.torchlike_index()
        self.online_cache[hook.name] = activations_dup
        return activations
    
    def receiver_hook(self, activations: Tensor, 
                            hook: HookPoint,
                            node: TLNodeIndex):
        cur_incoming_edge_type = get_incoming_edge_type(node)
        
        if cur_incoming_edge_type == TLEdgeType.DIRECT_COMPUTATION: 
            idx = node.torchlike_index()
            if not self.graph.reverse_graph[node]:
                activations[:][idx] = self.corrupted_cache[node.name][idx].to(self.device)
            return activations
        elif cur_incoming_edge_type == TLEdgeType.ADDITION:
            cur_idx = node.torchlike_index()
            activations[:][cur_idx] = self.corrupted_cache[node.name][cur_idx].to(self.device)
            
            for parent_node in self.graph.reverse_graph[node]:
                parent_idx = parent_node.torchlike_index()
                activations[cur_idx] += self.online_cache[parent_node.name][parent_idx].to(self.device)
                activations[cur_idx] -= self.corrupted_cache[parent_node.name][parent_idx].to(self.device)
            return activations
        else:
            return activations
        
    def add_sender_hook(self, node: TLNodeIndex, override=False):
        
        if not override and node.name in self.sender_hook_dict:
            # raise Exception(f"sender hook already exists {node}")
            return False
        
        self.model.add_hook(
            name=node.name, 
            hook=partial(self.sender_hook),
        )
        self.sender_hook_dict[node.name].add(node.  index)
        return True
    
    
    def add_all_sender_hooks(self):
        for node in self.graph.topo_order:
            if node.name not in self.sender_hook_dict:
                self.add_sender_hook(node)
        
        
        
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
        
        all_parent_nodes = sorted(self.graph.reverse_graph[cur_node].copy())
        for parent_node in all_parent_nodes:
            if cur_incoming_edge_type == TLEdgeType.ADDITION:
                self.add_sender_hook(parent_node)
                
            self.graph.remove_edge(parent_node, cur_node)
            old_eval = self.cur_eval
            new_eval = self.run_model_and_eval()
            print("try remove edge {} -> {}".format(cur_node, parent_node))
            print(f"new eval: {new_eval:.2f}, old eval: {old_eval:.2f}, diff: {(new_eval - old_eval):.2f}")
            if new_eval - old_eval < self.threshold:
                self.cur_eval = new_eval
            else: 
                self.graph.add_edge(parent_node, cur_node)
                if self.debug:
                    print(f"===== NODE NOT TRIMMED ====")
            
    def step(self):
        if self.current_node_idx >= len(self.graph.topo_order):
            print("no more nodes to process")
            return 
        cur_node = self.graph.topo_order[self.current_node_idx]
        print(f"{self.graph.reverse_graph[cur_node]=}")
        while self.current_node_idx > 0 and self.graph.node_disconnected(cur_node):
            self.graph.remove_node(cur_node)
            self.current_node_idx += 1
            cur_node = self.graph.topo_order[self.current_node_idx]
            self.cur_eval = self.run_model_and_eval()
        self.steps += 1

        
        self.cur_eval = self.run_model_and_eval()
        if self.debug:
            print(f"\n\n!!!! STEP {self.steps}: {cur_node} {self.current_node_idx}!!!! ")
            print(f"current eval: {self.cur_eval}")
            if self.steps == 1: 
                print(f"online cache keys: {self.online_cache.keys()}")
        
        cur_incoming_edge_type = get_incoming_edge_type(cur_node)
        if cur_incoming_edge_type != TLEdgeType.PLACEHOLDER:
            if self.debug:
                print(f"adding receiver hook {cur_node}")
            self.add_receiver_hook(cur_node)
            
        if cur_incoming_edge_type == TLEdgeType.DIRECT_COMPUTATION:
            self.add_sender_hook(cur_node, override=True)
        
        if cur_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"] \
            or cur_incoming_edge_type == TLEdgeType.PLACEHOLDER:
            pass
        else: 
            self.try_remove_edges(cur_node)
        
        self.current_node_idx += 1
        
            
        

        


