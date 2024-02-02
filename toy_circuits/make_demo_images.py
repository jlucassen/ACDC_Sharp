import graphviz 
import toy_circuits as tc

tree_size_param = 5
dag_size_param = 8
fan_in = 2

tree = tc.make_graph(2**tree_size_param-1, 2**tree_size_param, fan_in, input_value=True, tree=True)
tc.visualize_graph_components(tree, filename='circuits/tree')

dag = tc.make_graph(dag_size_param**2, dag_size_param, fan_in, input_value=True, tree=False)
tc.visualize_graph_components(dag, filename='circuits/dag')

tree_noise_component = tc.get_reachable_component(tree, noise=True)
tc.visualize_graph_components(tree, [tree_noise_component], filename='components/tree_noise_component')

tree_denoise_component = tc.get_reachable_component(tree, noise=False)
tc.visualize_graph_components(tree, [tree_denoise_component], filename='components/tree_denoise_component')

dag_noise_component = tc.get_reachable_component(dag, noise=True)
tc.visualize_graph_components(dag, [dag_noise_component], filename='components/dag_noise_component')

dag_denoise_component = tc.get_reachable_component(dag, noise=False)
tc.visualize_graph_components(dag, [dag_denoise_component], filename='components/dag_denoise_component')

tree_alternating_components = tc.alternating_components(tree, 5)
tc.visualize_graph_components(tree, tree_alternating_components, filename='alternating_components/tree_alternating_components')

dag_alternating_components = tc.alternating_components(dag, 5)
tc.visualize_graph_components(dag, dag_alternating_components, filename='alternating_components/dag_alternating_components')