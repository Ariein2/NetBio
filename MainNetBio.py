# %% THRESHOLD
import networkx as nx
import igraph 
import numpy as np 
import ndlib
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.bokeh.MultiPlot import MultiPlot
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence
import matplotlib as plot
import hvplot.networkx as hvnx
from itertools import compress 
import pandas as pd
import random 
#%%
# '''
# # Random graph:
# av_degree= 10
# prob = 2*av_degree/1000
# G = nx.erdos_renyi_graph(1000, prob)
# av = G.number_of_edges()/G.number_of_nodes()
# print(av)
# nx.draw(G)
# nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
# '''

# LOAD GRAPH: 
data=nx.read_graphml('budapest_large.graphml')
shell= hvnx.draw(data)
data.remove_nodes_from(list(nx.isolates(data)))
shell2= hvnx.draw(data)
#shell+shell2

# Find High degree nodes
data_filtered = data 
percent_removed = 0.5  
degree = list(dict(data_filtered.degree).values())

# Select nodes from top degree quartile: 
quartile = np.quantile(degree, 0.75) 
keys = list(dict(data_filtered .degree).keys())
bool_quart = degree >= quartile
quartile_keys = list(compress(keys, bool_quart))
remove_nodes = random.sample(quartile_keys, round(percent_removed*len(quartile_keys)))
edge_remove = list(data_filtered.edges(remove_nodes))
data_filtered.remove_edges_from(edge_remove)

#data_filtered.remove_nodes_from(remove_nodes)
draw = hvnx.draw(data_filtered)
draw 
# nx.draw(data)

#%% # Model selection
multi = MultiPlot()
model = ep.ThresholdModel(data)

# # Model Configuration
config = mc.Configuration()
config.add_model_parameter('fraction_infected', 0.01)

# Setting node parameters
threshold = 0.15
for i in data.nodes():
    config.add_node_configuration("threshold", i, threshold)

model.set_initial_status(config)

# Simulation execution
iterations = model.iteration_bunch(10)
trends = model.build_trends(iterations)

viz = DiffusionTrend(model, trends)
p = viz.plot(width=400, height=400)
multi.add_plot(p)

viz2 = DiffusionPrevalence(model, trends)
p2 = viz2.plot(width=400, height=400)
multi.add_plot(p2)
show(multi.plot())


# %%
import networkx as nx
import random
import ndlib
import numpy as np
import matplotlib.pyplot as plt
from ndlib.models.compartments.enums.NumericalType import NumericalType
from ndlib.models.ContinuousModel import ContinuousModel
from ndlib.models.compartments.NodeStochastic import NodeStochastic

import ndlib.models.ModelConfig as mc

################### MODEL SPECIFICATIONS ###################

constants = {
    'q': 0.8,
    'b': 0.5,
    'd': 0.2,
    'h': 0.2,
    'k': 0.25,
    'S+': 0.5,
}
constants['p'] = 2*constants['d']

def initial_v(node, graph, status, constants):
    return min(1, max(0, status['C']-status['S']-status['E']))

def initial_a(node, graph, status, constants):
    return constants['q'] * status['V'] + (np.random.poisson(status['lambda'])/7)

initial_status = {
    'C': 0,
    'S': constants['S+'],
    'E': 1,
    'V': initial_v,
    'lambda': 0.5,
    'A': initial_a
}

def update_C(node, graph, status, attributes, constants):
    return status[node]['C'] + constants['b'] * status[node]['A'] * min(1, 1-status[node]['C']) - constants['d'] * status[node]['C']

def update_S(node, graph, status, attributes, constants):
    return status[node]['S'] + constants['p'] * max(0, constants['S+'] - status[node]['S']) - constants['h'] * status[node]['C'] - constants['k'] * status[node]['A']

def update_E(node, graph, status, attributes, constants):
    # return status[node]['E'] - 0.015 # Grasman calculation

    avg_neighbor_addiction = 0
    for n in graph.neighbors(node):
        avg_neighbor_addiction += status[n]['A']

    return max(-1.5, status[node]['E'] - avg_neighbor_addiction / 50) # Custom calculation

def update_V(node, graph, status, attributes, constants):
    return min(1, max(0, status[node]['C']-status[node]['S']-status[node]['E']))

def update_lambda(node, graph, status, attributes, constants):
    return status[node]['lambda'] + 0.01

def update_A(node, graph, status, attributes, constants):
    return constants['q'] * status[node]['V'] + min((np.random.poisson(status[node]['lambda'])/7), constants['q']*(1 - status[node]['V']))

################### MODEL CONFIGURATION ###################

# Network definition
g = nx.random_geometric_graph(200, 0.125)

# Visualization config
visualization_config = {
    'plot_interval': 2,
    'plot_variable': 'A',
    'variable_limits': {
        'A': [0, 0.8],
        'lambda': [0.5, 1.5]
    },
    'show_plot': True,
    'plot_output': './c_vs_s.gif',
    'plot_title': 'Self control vs craving simulation',
}

# Model definition
craving_control_model = ContinuousModel(g, constants=constants)
craving_control_model.add_status('C')
craving_control_model.add_status('S')
craving_control_model.add_status('E')
craving_control_model.add_status('V')
craving_control_model.add_status('lambda')
craving_control_model.add_status('A')

# Compartments
condition = NodeStochastic(1)

# Rules
craving_control_model.add_rule('C', update_C, condition)
craving_control_model.add_rule('S', update_S, condition)
craving_control_model.add_rule('E', update_E, condition)
craving_control_model.add_rule('V', update_V, condition)
craving_control_model.add_rule('lambda', update_lambda, condition)
craving_control_model.add_rule('A', update_A, condition)

# Configuration
config = mc.Configuration()
craving_control_model.set_initial_status(initial_status, config)
craving_control_model.configure_visualization(visualization_config)

################### SIMULATION ###################

# Simulation
iterations = craving_control_model.iteration_bunch(100, node_status=True)
trends = craving_control_model.build_trends(iterations)

################### VISUALIZATION ###################

# Show the trends of the model
craving_control_model.plot(trends, len(iterations), delta=True)

# Recreate the plots shown in the paper to verify the implementation
x = np.arange(0, len(iterations))
plt.figure()

plt.subplot(221)
plt.plot(x, trends['means']['E'], label='E')
plt.plot(x, trends['means']['lambda'], label='lambda')
plt.legend()

plt.subplot(222)
plt.plot(x, trends['means']['A'], label='A')
plt.plot(x, trends['means']['C'], label='C')
plt.legend()

plt.subplot(223)
plt.plot(x, trends['means']['S'], label='S')
plt.plot(x, trends['means']['V'], label='V')
plt.legend()

plt.show()

# Show animated plot
craving_control_model.visualize(iterations)
# %%
import networkx as nx
import igraph 
from ndlib.models.compartments.NodeStochastic import NodeStochastic
import numpy as np 
import ndlib
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.bokeh.MultiPlot import MultiPlot
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence
import ndlib.models.compartments.NodeThreshold as NodeThreshold
import matplotlib as plot
import hvplot.networkx as hvnx
from ndlib.models.ContinuousModel import ContinuousModel

g = nx.erdos_renyi_graph(1000, 0.3)

g.edges
constants = {
    'fraction_infected' : 0.01,
    'threshold' : 0.15
}

def initial_i (node, graph, status, constants):

    if (node > 500):
        return 1
    else: 
        return 0

def initial_a (node, graph, status, constants):
    if (node <= 500):
        return 1
    else: 
        return 0

initial_status = {
        'Inactive': initial_i,
        'Active': initial_a,
}

model = ContinuousModel(g)

model.add_status('Inactive')
model.add_status('Active')

def update (node, graph, status, attributes, constants):
    av_state = 0
    for n in graph.neighbors(node):
        av_state = sum(av_state, status[node]['Active'])
    neig_score = av_state/len(graph.neighbors[node])
    if neigh_score >= threshold: 
        return status[node]['Active'] == 1, status[node]['Inactive'] == 0

c1 = NodeStochastic(1) 
#c1 = NodeThreshold(0.15, triggering_status= "Active")
model.add_rule("Inactive", update, c1)

config = mc.Configuration()
#config.add_model_parameter('fraction_infected', 0.1)
model.set_initial_status(initial_status, config)

iterations = model.iteration_bunch(10, node_status=True)


# Visualization config
visualization_config = {
    'plot_interval': 1,
    'plot_variable': 'Active',
    'variable_limits': {
        'Active': [0, 1]
    },
    'show_plot': True,
    'plot_output': './model_animation.gif',
    'plot_title': 'Animated network',
}

model.configure_visualization(visualization_config)
model.visualize(iterations)
# %%
