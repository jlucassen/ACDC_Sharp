import graphviz 
import toy_circuits as tc
from copy import deepcopy

def visualize_graph_edges(graph:list, edge_batches:list=[{}], filename:str='temp'):
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    for i, edges in enumerate(edge_batches):
        viz = graphviz.Digraph(engine='dot')
        for node in graph:
            node_kwargs = {
                'fillcolor':color_code[node.gate],
                'style':'filled',
                'fontcolor':'white',
            }
            viz.node(str(node.id), **node_kwargs)
            for child in node.children:
                edge_key = (str(child.id), str(node.id))
                if edge_key in edges.keys():
                    viz.edge(str(child.id), str(node.id), **edges[edge_key]) # plot selected edges with corresponding kwargs
                else:
                    viz.edge(str(child.id), str(node.id))
        viz.render(f'toy_circuits/circuit_viz/{filename}_{i}.gv')

def get_edge_grad(node, child, noise):
    old_forward = node.forward()
    saved_gate = child.gate
    saved_value = child.value
    child.gate = 'INPUT'
    child.value = not noise
    new_forward = node.forward()
    child.gate = saved_gate
    child.value = saved_value
    return new_forward != old_forward

def get_all_edge_grads(graph:list, noise=True, color='red'):
    edges = {}
    for node in graph:
        for child in node.children:
            if get_edge_grad(node, child, noise):
                edges[(str(child.id), str(node.id))] = {'color':color} # flipping child can change node, mark edge
    return edges

def set_all_inputs(graph:list, value:bool=False):
    for node in graph:
        if node.gate == 'INPUT':
            node.value = value

def recursive_add(node, edges, combined_edges):
    nodes_to_add = []
    for child in node.children:
        edge_key = (str(child.id), str(node.id))
        if edge_key in edges.keys():
            nodes_to_add.append(child)
            nodes_to_add += recursive_add(child, edges, combined_edges)
            combined_edges[edge_key] = edges[edge_key]
    return nodes_to_add

def frankenstein_path_algorithm(graph:list, n_iters:int=5, greedy=False):
    noising_edges = get_all_edge_grads(graph, noise=True, color='red')
    set_all_inputs(graph, False)
    denoising_edges = get_all_edge_grads(graph, noise=False, color='blue')

    combined_edges = {}
    included_nodes = [graph[-1]]
    edge_batches = []
    for _ in range(n_iters):
        print(f"{[n.id for n in included_nodes]=}")
        nodes_to_include = []
        for node in included_nodes:
            print(f"{node.id=}, {[n.id for n in node.children]=}")
            for child in node.children:
                if child not in included_nodes:
                    edge_key = (str(child.id), str(node.id))
                    if edge_key in noising_edges.keys():
                        combined_edges[edge_key] = noising_edges[edge_key]
                        nodes_to_include.append(child) # we have now found a path from child up to exit that has high grads all the way
                        if greedy:
                            nodes_to_include += recursive_add(child, noising_edges, combined_edges) # adds to combined edges internally
                        if edge_key in denoising_edges.keys():
                            raise NotImplementedError # don't know how to handle doubles yet
                    elif edge_key in denoising_edges:
                        combined_edges[edge_key] = denoising_edges[edge_key]
                        nodes_to_include.append(child) # we have now found a path from child up to exit that has high grads all the way
                        if greedy:
                            nodes_to_include += recursive_add(child, denoising_edges, combined_edges) # adds to combined edges internally
        print(f"{[n.id for n in nodes_to_include]=}")
        included_nodes += nodes_to_include
        edge_batches.append(combined_edges.copy())
    return edge_batches
                

tree_size_param = 5
dag_size_param = 8
fan_in = 2

tree_on = tc.make_graph(2**tree_size_param-1, 2**tree_size_param, fan_in, input_value=True, tree=True)
tree_off = tc.make_graph(2**tree_size_param-1, 2**tree_size_param, fan_in, input_value=False, tree=True)

tree_on_noise_edges = get_all_edge_grads(tree_on, noise=True)
tree_on_denoise_edges = get_all_edge_grads(tree_on, noise=False, color='blue')
tree_off_noise_edges = get_all_edge_grads(tree_off, noise=True)
tree_off_denoise_edges = get_all_edge_grads(tree_off, noise=False, color='blue')

#visualize_graph_edges(tree_on, [tree_on_noise_edges], 'tree_on_noise_edges')
#visualize_graph_edges(tree_on, [tree_on_denoise_edges], 'tree_on_denoise_edges')
#visualize_graph_edges(tree_off, [tree_off_noise_edges], 'tree_off_noise_edges')
#visualize_graph_edges(tree_off, [tree_off_denoise_edges], 'tree_off_denoise_edges')

tree_on_franken_nongreed_edges = frankenstein_path_algorithm(deepcopy(tree_on), n_iters=5)
tree_on_franken_greed_edges = frankenstein_path_algorithm(deepcopy(tree_on), n_iters=5, greedy=True)
print(tree_on_franken_greed_edges)
#visualize_graph_edges(tree_on, tree_on_franken_nongreed_edges, filename='tree_frankenstein_nongreedy_edges')
visualize_graph_edges(tree_on, tree_on_franken_greed_edges, filename='tree_frankenstein_greedy_edges')