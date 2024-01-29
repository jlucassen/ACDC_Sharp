import toy_circuits
import graphviz

def check_noising_reachability(graph:list):
    assert all([node.children == True for node in graph if node.gate == 'INPUT'])
    assert graph[-1].forward() == True
    reachability_dict = {}
    for node in graph:
        saved_gate = node.gate
        saved_children = node.children
        node.gate = 'INPUT'
        node.children = False
        reachability_dict[node.id] = not graph[-1].forward() # check if we corrupt output
        node.gate = saved_gate
        node.children = saved_children
    return reachability_dict

def check_denoising_reachability(graph:list):
    assert all([node.children == False for node in graph if node.gate == 'INPUT'])
    assert graph[-1].forward() == False
    reachability_dict = {}
    for node in graph:
        saved_gate = node.gate
        saved_children = node.children
        node.gate = 'INPUT'
        node.children = True
        reachability_dict[node.id] = graph[-1].forward() # check if we un-corrupt output
        node.gate = saved_gate
        node.children = saved_children
    return reachability_dict

def visualize_reachability(graph:list, reachability:dict, filename:str='temp.gv'):
    viz = graphviz.Digraph(engine='dot')
    color_code = {'INPUT':'black', 'AND':'red', 'OR':'blue'}
    # layer_counts = [0]*len(graph)
    for node in graph:
        plot_kwargs = {
            'fillcolor':color_code[node.gate],
            'style':'filled',
            'fontcolor':'white',
            #'pos':f"{count_to_pos(layer_counts[node.layer])},{node.layer}!"
        }
        if reachability[node.id]:
            plot_kwargs['penwidth'] = '5'
            plot_kwargs['color'] = 'yellow'
        viz.node(str(node.id), **plot_kwargs)
        # layer_counts[node.layer] += 1
        if node.gate != 'INPUT':
            for child in node.children:
                viz.edge(str(child.id), str(node.id))
    viz.render(f'toy_circuits/circuit_viz/{filename}')

def make_demo_images(save=False):
    noising_tree = toy_circuits.make_tree(15, 16, 2, True)
    noising_tree_reach_dict = check_noising_reachability(noising_tree)
    print(sum(noising_tree_reach_dict.values()))
    visualize_reachability(noising_tree, noising_tree_reach_dict, 'noising_tree_demo.gz' if save else 'temp.gz')

    denoising_tree = toy_circuits.make_tree(15, 16, 2, False)
    denoising_tree_reach_dict = check_denoising_reachability(denoising_tree)
    print(sum(denoising_tree_reach_dict.values()))
    visualize_reachability(denoising_tree, denoising_tree_reach_dict, 'denoising_tree_demo.gz' if save else 'temp.gz')

    noising_dag = toy_circuits.make_dag(15, 16, 2, True)
    noising_dag_reach_dict = check_noising_reachability(noising_dag)
    print(sum(noising_dag_reach_dict.values()))
    visualize_reachability(noising_dag, noising_dag_reach_dict, 'noising_dag_demo.gz' if save else 'temp.gz')

    denoising_dag = toy_circuits.make_dag(15, 16, 2, False)
    denoising_dag_reach_dict = check_denoising_reachability(denoising_dag)
    print(sum(denoising_dag_reach_dict.values()))
    visualize_reachability(denoising_dag, denoising_dag_reach_dict, 'denoising_dag_demo.gz' if save else 'temp.gz')

make_demo_images(save=False)