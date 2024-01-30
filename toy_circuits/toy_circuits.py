import numpy as np
import graphviz 

class Node:
    def __init__(self, id:int, gate:str, children:list|bool):
        assert (gate == 'AND') | (gate == 'OR') | (gate == 'INPUT')
        assert (isinstance(children, list) and gate != 'INPUT') or (isinstance(children, bool) and gate == 'INPUT')
        self.id = id
        self.gate = gate
        self.children = children
        self.layer = 0
        if isinstance(children, list):
            self.layer = 1 + max([child.layer for child in self.children])
        
    def forward(self):
        if self.gate == 'AND':
            return all([child.forward() for child in self.children])
        elif self.gate == 'OR':
            return any([child.forward() for child in self.children])
        elif self.gate == 'INPUT':
            return self.children
        
    def __repr__(self):
        if self.gate == 'INPUT':
            return f"{self.id=}, {self.gate=}, {self.children=}"
        else:
            return f"{self.id=}, {self.gate=}, {[child.id for child in self.children]=}"

def make_graph(num_gates:int, num_inputs:int, fan_in:int,  input_value:bool, tree=True):
    graph = []
    childless = []
    count = 0
    if tree: assert num_gates == num_inputs - 1 and (num_inputs & num_gates == 0) and num_inputs != 0 # bit method to check if num_inputs is power of 2
    for _ in range(num_inputs):
        new_node = Node(count, 'INPUT', input_value)
        graph.append(new_node)
        childless.append(new_node)
        count += 1
    for _ in range(num_gates):
        if tree:
            children = list(np.random.choice(childless, size=fan_in, replace=False)) # sample from childless, to prevent multi outputs
        else:
            p = [1/(len(node.children)*1000+1) if isinstance(node.children, list) else 1e-4 for node in graph]# bias choice to get some depth
            p = p / np.sum(p)
            children = list(np.random.choice(graph, size=fan_in, replace=False, p=p)) # sample from graph to allow multi outputs. 
        new_node = Node(
            count,
            gate=np.random.choice(['AND', 'OR']),
            children=children
            )
        graph.append(new_node)
        childless.append(new_node)
        for child in children:
            if child in childless:
                childless.remove(child) # make sure to never grab the same wire as child twice
        count += 1
    if not tree:
        graph.append(Node(
            count,
            gate=np.random.choice(['AND', 'OR']),
            children=childless
            ))
    return graph

def get_reachable_component(graph:list, sensor_list:list=None, noise=True):
    if sensor_list is None: sensor_list = [graph[-1].id] # if no sensor give, assume last node is sensor
    
    for node in graph:
        if node.gate == 'INPUT':
            node.children = noise # set up for noising/denoising

    assert all([node.children == noise for node in graph if node.gate == 'INPUT'])
    assert graph[-1].forward() == noise
    component_nodes = []

    for node in graph:
        saved_gate = node.gate
        saved_children = node.children
        node.gate = 'INPUT'
        node.children = not noise # noise
        for sensor in sensor_list:
            if not graph[sensor].forward() == noise: # check if any sensor flips
                component_nodes.append(node.id)
                break
        node.gate = saved_gate
        node.children = saved_children
    return component_nodes

def alternating_components(graph:list, n_iters:int):
    components = [get_reachable_component(graph)]
    for i in range(n_iters-1):
        components.append(get_reachable_component(graph, components[-1], noise = bool(i%2)))
    return components

def visualize_graph_components(graph:list, components:list=[[]], filename:str='temp.gv'):
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    for i, component in enumerate(components):
        viz = graphviz.Digraph(engine='dot')
        for node in graph:
            plot_kwargs = {
                'fillcolor':color_code[node.gate],
                'style':'filled',
                'fontcolor':'white',
            }
            if node.id in component:
                plot_kwargs['penwidth'] = '5'
                plot_kwargs['color'] = 'yellow'
            viz.node(str(node.id), **plot_kwargs)
            if node.gate != 'INPUT':
                for child in node.children:
                    viz.edge(str(child.id), str(node.id))
        viz.render(f'toy_circuits/circuit_viz/{filename}_{i}')