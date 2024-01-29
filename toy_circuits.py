import random
import graphviz 

class Node:
    def __init__(self, id:int, gate:str, children:list|bool):
        assert (gate == 'AND') | (gate == 'OR') | (gate == 'INPUT')
        assert (isinstance(children, list) and gate != 'INPUT') or (isinstance(children, bool) and gate == 'INPUT')
        self.id = id
        self.gate = gate
        self.children = children
        
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

def visualize_graph(graph:list):
    viz = graphviz.Digraph()
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    for node in graph:
        viz.node(str(node.id), color=color_code[node.gate])
        if node.gate != 'INPUT':
            for child in node.children:
                viz.edge(str(child.id), str(node.id))
    viz.render('doctest-output/round-table.gv')

visualize_graph(make_tree(2, 3, 2, False))