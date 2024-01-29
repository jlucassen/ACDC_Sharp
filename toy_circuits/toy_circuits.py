import random
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
        
def make_dag(num_gates:int, num_inputs:int, fan_in:int,  input_value:bool):
    graph = []
    count = 0
    for _ in range(num_inputs):
        graph.append(Node(count, 'INPUT', input_value))
        count += 1
    for _ in range(num_gates):
        graph.append(Node(
            count,
            gate=random.choice(['AND', 'OR']),
            children=random.choices(graph, k=fan_in)
            ))
        count += 1
    return graph

def make_tree(num_gates:int, num_inputs:int, fan_in:int,  input_value:bool):
    graph = []
    able_to_child = []
    count = 0
    for _ in range(num_inputs):
        new_node = Node(count, 'INPUT', input_value)
        graph.append(new_node)
        able_to_child.append(new_node)
        count += 1
    for _ in range(num_gates):
        children=random.sample(able_to_child, k=fan_in) # sample without replacement
        new_node = Node(
            count,
            gate=random.choice(['AND', 'OR']),
            children=children
            )
        graph.append(new_node)
        able_to_child.append(new_node)
        for child in children:
            able_to_child.remove(child) # make sure to never grab the same wire as child twice
        count += 1
    return graph

def visualize_graph(graph:list, filename:str='temp.gv'):
    def count_to_pos(count:int):
        return count * (-1 if count%2==0 else 1)

    viz = graphviz.Digraph(engine='dot')
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    layer_counts = [0]*len(graph)
    for node in graph:
        if node.gate != 'INPUT':
            viz.node(str(node.id), color=color_code[node.gate])#, pos=f"{count_to_pos(layer_counts[node.layer])},{node.layer}!")
            layer_counts[node.layer] += 1
            for child in node.children:
                viz.edge(str(child.id), str(node.id))
        else:
            viz.node(str(node.id), color=color_code[node.gate])#, pos=f"{count_to_pos(layer_counts[node.layer])},{node.layer}!")
            layer_counts[node.layer] += 1
    viz.render(f'toy_circuits/circuit_viz/{filename}')

visualize_graph(make_tree(15, 16, 2, False), 'demo-tree.gv')
visualize_graph(make_dag(32, 16, 2, False), 'demo-dag.gv')