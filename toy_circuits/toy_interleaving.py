import graphviz 
import toy_circuits as tc

def visualize_graph_edges(graph:list, edges:dict={}, filename:str='temp.gv'):
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
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
            if edge_key in edges:
                viz.edge(str(child.id), str(node.id), **edges[edge_key]) # plot selected edges with corresponding kwargs
            else:
                viz.edge(str(child.id), str(node.id))
    viz.render(f'toy_circuits/circuit_viz/{filename}')

def get_node_to_node_derivatives(graph:list):
    edges = {}
    for node in graph:
        for child in node.children:
            edges[(node.id, child.id)] = {'color':'red'}
    return edges

tree_size_param = 5
dag_size_param = 8
fan_in = 2

tree = tc.make_graph(2**tree_size_param-1, 2**tree_size_param, fan_in, input_value=True, tree=True)
edges = get_node_to_node_derivatives(tree)
visualize_graph_edges(tree, edges)