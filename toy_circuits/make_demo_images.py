import graphviz 
import toy_circuits as tc

tree_size_param = 5
dag_size_param = 10
fan_in = 2

tree = tc.make_graph(2**tree_size_param-1, 2**tree_size_param, fan_in, input_value=True, tree=True)
tc.visualize_graph_components(tree, filename='tree.gv')

dag = tc.make_graph(dag_size_param**2, dag_size_param, fan_in, input_value=True, tree=False)
tc.visualize_graph_components(dag, filename='dag.gv')

tree_noise_component = tc.get_reachable_component(tree, noise=True)
tc.visualize_graph_components(tree, [tree_noise_component], filename='tree_noise_component.gv')

tree_denoise_component = tc.get_reachable_component(tree, noise=False)
tc.visualize_graph_components(tree, [tree_denoise_component], filename='tree_denoise_component.gv')

dag_noise_component = tc.get_reachable_component(dag, noise=True)
tc.visualize_graph_components(dag, [dag_noise_component], filename='dag_noise_component.gv')

dag_denoise_component = tc.get_reachable_component(dag, noise=False)
tc.visualize_graph_components(dag, [dag_denoise_component], filename='dag_denoise_component.gv')

tree_alternating_components = tc.alternating_components(tree, 5)
tc.visualize_graph_components(tree, tree_alternating_components, filename='tree_alternating_components.gv')

dag_alternating_components = tc.alternating_components(dag, 5)
tc.visualize_graph_components(dag, dag_alternating_components, filename='dag_alternating_components.gv')