from transformer_lens.HookedTransformer import HookedTransformer

from graph_builder import TLGraph

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
        self.current_node = self.graph.topo_order[-1]
        clean_logits, self.clean_cache = model.run_with_cache(self.clean_ds)
        corr_logits, self.corr_cache = model.run_with_cache(self.corr_ds)

        # put this in init, since we're not doing the path-dependent thing
        self.clean_metric_value = self.metric(clean_logits)
        self.corr_metric_value = self.metric(corr_logits)
    
    def patch_hook(self, hook_point):
        activations_to_patch = self.corr_cache[hook_point]
        self.model.hook_dict[hook_point][:] = activations_to_patch
        return self.model.hook_dict[hook_point]
        
    def freeze_hook(self, hook_point):
        activations_to_patch = self.corr_cache[hook_point]
        self.model.hook_dict[hook_point][:] = activations_to_patch
        return self.model.hook_dict[hook_point]

    def step(self):
        if self.current_node is None:
            return
        
        for sender in self.graph.backwards_graph[self.current_node]:
            # do patch
            self.graph.topo_order.index(sender)
            self.graph.topo_order
            heads_to_freeze = self.graph.topo_order[self.graph.topo_order.index(sender):self.graph.topo_order(self.current_node)] # between sender and current node
            patched_logits = self.model.run_with_hooks(fwd_hooks=[
                *[(head, self.freeze_hook) for head in heads_to_freeze]
                (sender.hook_point, self.patch_hook),
            ])
            patched_metric_value = self.metric(patched_logits)
            patch_effect =  patched_metric_value - self.clean_metric_value
            relative_patch_effect = patch_effect / (self.clean_metric_value - self.corr_metric_value)
        





