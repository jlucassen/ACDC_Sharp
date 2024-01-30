import toy_circuits
import graphviz

def get_noising_component(graph:list, sensor_list:list):
    for node in graph:
        if node.gate == 'INPUT':
            node.children = True # set up for noising

    assert all([node.children == True for node in graph if node.gate == 'INPUT'])
    assert graph[-1].forward() == True
    component_dict = {}

    for node in graph:
        saved_gate = node.gate
        saved_children = node.children
        node.gate = 'INPUT'
        node.children = False # noise
        component_dict[node.id] = False
        for sensor in sensor_list:
            if sensor_list[sensor] and graph[sensor].forward() == False: # check if any sensor flips
                component_dict[node.id] = True
        node.gate = saved_gate
        node.children = saved_children
    return component_dict

def get_denoising_component(graph:list, sensor_list:list):
    for node in graph:
        if node.gate == 'INPUT':
            node.children = False # set up for denoising

    assert all([node.children == False for node in graph if node.gate == 'INPUT'])
    assert graph[-1].forward() == False
    component_dict = {}

    for node in graph:
        saved_gate = node.gate
        saved_children = node.children
        node.gate = 'INPUT'
        node.children = True # noise
        component_dict[node.id] = False
        for sensor in sensor_list:
            if sensor_list[sensor] and graph[sensor].forward() == True: # check if any sensor flips
                component_dict[node.id] = True
        node.gate = saved_gate
        node.children = saved_children
    return component_dict

def alternating_components(graph:list, n_iters:int):
    components = [dict([(node.id, False) for node in graph])]
    components[0][graph[-1].id] = True # set output as initial sensor
    for _  in range(n_iters):
        components.append(get_noising_component(graph, components[-1]))
        components.append(get_denoising_component(graph, components[-1]))
    return components

def visualize_components(graph:list, components:list, filename:str='temp.gv'):
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    for i, component in enumerate(components):
        viz = graphviz.Digraph(engine='dot')
        for node in graph:
            plot_kwargs = {
                'fillcolor':color_code[node.gate],
                'style':'filled',
                'fontcolor':'white',
            }
            if component[node.id]:
                plot_kwargs['penwidth'] = '5'
                plot_kwargs['color'] = 'yellow'
            viz.node(str(node.id), **plot_kwargs)
            if node.gate != 'INPUT':
                for child in node.children:
                    viz.edge(str(child.id), str(node.id))
        viz.render(f'toy_circuits/circuit_viz/{filename}_{i}')

tree = toy_circuits.make_tree(15, 16, 2, True)
components = alternating_components(tree, n_iters = 3)
print(components)
visualize_components(tree, components, 'temp.gz')