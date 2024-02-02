from transformer_lens.HookedTransformer import HookedTransformer, HookPoint
from TLComponentGraph import TLGraph, get_incoming_edge_type, TLNodeIndex, TLEdgeType
from collections import OrderedDict, defaultdict, OrderedSet
from functools import partial 
from torch import Tensor
import torch as t
from enum import Enum
class TLExperimentMode(Enum):
    NOISING = 0
    DENOISING = 1

    def __eq__(self, other):
        if type(self).__qualname__ != type(other).__qualname__:
            return NotImplemented
        return self.name == other.name and self.value == other.value

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
        self.mode = TLExperimentMode.DENOISING
        self.frontier = OrderedSet([self.graph.topo_order[0]])
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
        self.online_cache[hook.name] = activations_dup
        return activations
    
    def receiver_hook(self, activations: Tensor, 
                            hook: HookPoint,
                            node: TLNodeIndex):
        cur_incoming_edge_type = get_incoming_edge_type(node)
        if cur_incoming_edge_type == TLEdgeType.DIRECT_COMPUTATION: 
            idx = node.torchlike_index()
            parent_node = next(iter(self.graph.reverse_graph[node]))
            edge = self.graph.edges[parent_node][node]
            if not edge.present: 
                print(f"DIRECT_COMPUTATION corrupting edge {node}")
                activations[:][idx] = self.corrupted_cache[node.name][idx].to(self.device)
            return activations
        elif cur_incoming_edge_type == TLEdgeType.ADDITION:
            cur_idx = node.torchlike_index()
            activations[:][cur_idx] = self.corrupted_cache[node.name][cur_idx].to(self.device)
            for parent_node in self.graph.reverse_graph[node]:
                parent_idx = parent_node.torchlike_index()
                edge = self.graph.edges[parent_node][node]
                if self.mode == TLExperimentMode.NOISING and not edge.present:
                    continue
                elif self.mode == TLExperimentMode.DENOISING and edge.present:
                    continue
                
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
            if parent_node not in self.frontier:
                self.frontier.add(parent_node)
            
            edge = self.graph.edges[parent_node][cur_node]
            print(f"try {'denoise' if self.mode == TLExperimentMode.DENOISING else 'noising'} edge {cur_node} -> {parent_node}")
            edge.present = False
            old_eval = self.cur_eval
            new_eval = self.run_model_and_eval()
            
            print(f"new eval: {new_eval:.2f}, old eval: {old_eval:.2f}, diff: {(new_eval - old_eval):.2f}")
            if self.pass_eval_metric(old_eval, new_eval):
                self.cur_eval = new_eval
            else: 
                edge.present = True
                if self.debug:
                    print(f"===== NODE NOT TRIMMED ====")
            
    def step(self):
        # if self.current_node_idx >= len(self.graph.topo_order):
        #     print("no more nodes to process")
        #     return 
        # cur_node = self.graph.topo_order[self.current_node_idx]
        # while self.current_node_idx > 0 and self.graph.node_disconnected(cur_node):
        #     self.graph.remove_node(cur_node)
        #     self.current_node_idx += 1
        #     cur_node = self.graph.topo_order[self.current_node_idx]
        #     self.cur_eval = self.run_model_and_eval()
        # self.steps += 1
        self.mode = TLExperimentMode.DENOISING if self.mode == TLExperimentMode.NOISING else TLExperimentMode.NOISING
        self.cur_eval = self.run_model_and_eval()
        
        for cur_node in self.frontier: 
            if cur_node.visited[self.mode]:
                raise Exception(f"node {cur_node} already visited in mode {self.mode}")
            cur_node.visited[self.mode] = True
            
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
                
            if all(cur_node.visited[self.mode]):
                self.frontier.remove(cur_node)
        
        # self.current_node_idx += 1
        
            
        

        


