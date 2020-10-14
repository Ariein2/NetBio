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
import selenium
from itertools import compress 
import pandas as pd
import random 
from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison
from py2cytoscape.data.cyrest_client import CyRestClient
# 'conda install -c conda-forge firefox geckodriver' 

#%% PREPROCESSING 

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
data = nx.read_graphml('budapest_large.graphml')
len(data.edges())
main_data = data 
shell = hvnx.draw(data, node_color = 'lightblue')
main_data.remove_nodes_from(list(nx.isolates(main_data)))
len(data.nodes())
'''
for node in data.nodes:
    if (data.nodes[node]["dn_hemisphere"] == "left"):
        data.nodes[node]["color"] = "ligthblue"
    else:
        data.nodes[node]["color"] = "darksalmon"
'''
shell2 = hvnx.draw(main_data, node_color = 'lightblue')
shell2 

#cy = CyRestClient()
#network = cy.network.create_from_networkx(data)

#layout =shell+shell2
#layout
#hvnx.save(layout,'layout.png')

#output_file(shell)
#%% HIGH DEGREE NODES 

# Find High degree nodes
data_filtered = main_data 
hvnx.draw(data_filtered)
nodes_removed =10
degree = list(dict(data_filtered.degree).values())

# Select nodes from top degree quartile: 
quartile = np.quantile(degree, 0.75) 
keys = list(dict(data_filtered .degree).keys())
bool_quart = degree >= quartile
quartile_keys = list(compress(keys, bool_quart))
selected_nodes = random.sample(quartile_keys,nodes_removed)
edge_remove = list(data_filtered.edges(selected_nodes))
data_filtered.remove_edges_from(edge_remove)

#data_filtered.remove_nodes_from(remove_nodes)
draw = hvnx.draw(data_filtered)
draw 
# %% VULNERABLE NODES 

# Find vulnerable nodes
vulnerable_net = data
nodes_removed = 10
degree = list(dict(vulnerable_net.degree).values())
influence =[]
for deg in degree: 
    if (deg == 0) : 
        influence.append(0) 
    else :
        influence.append(1/deg)

#influence = [1 /deg for deg in list(dict(vulneable_net.degree).values())]
threshold = np.float64(0.15)
bool_inf = influence >= threshold

# Select nodes from top degree quartile: 
keys = list(dict(vulnerable_net.degree).keys())
vulnerable_keys = list(compress(keys, bool_inf))
selected_nodes = random.sample(vulnerable_keys, nodes_removed)
edge_remove = list(vulnerable_net.edges(selected_nodes))
vulnerable_net.remove_edges_from(edge_remove)

#data_filtered.remove_nodes_from(remove_nodes)
draw = hvnx.draw(vulnerable_net)
draw 
# nx.draw(data)

#%% Alzheimer related area
alzh_net = main_data

node_info= pd.DataFrame(alzh_net.nodes._nodes)
areas = node_info.loc[['dn_fsname']]
nodes_removed = 10
area_bool = []
for index, text in areas.iteritems() :
    string= str(text)
    result = string.find ('Hippocampus')
    result2 = string.find ('middletemporal')
    if (result == -1 & result2 == -1):
        area_bool.append(False)
    else: 
        area_bool.append(True)

keys = list(dict(alzh_net.degree).keys())
alzh_keys = list(compress(keys, area_bool))
selected_nodes = random.sample(alzh_keys, nodes_removed)
edge_remove = list(alzh_net.edges(selected_nodes))
alzh_net.remove_edges_from(edge_remove)
hvnx.draw(alzh_net)

#%% # Model selection

def model (data, frac_inf, threshold, iter):
    multi = MultiPlot()
    model = ep.ThresholdModel(data)
    # # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter('fraction_infected', frac_inf)

    # Setting node parameters
    for i in data.nodes():
        config.add_node_configuration("threshold", i, threshold)

    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(iter)
    trends = model.build_trends(iterations)
    return [model, iterations,trends]

[mod_norm, iter_normal, trends_normal] = model(main_data, 0.01, 0.15, 20)
[mod_degree, iter_degree, trends_degree] = model(data_filtered, 0.01, 0.15, 20)
[mod_vul, iter_vul, trends_vul] = model(vulnerable_net, 0.01, 0.15, 20)
[mod_alzh, iter_alzh, trends_alzh] = model(alzh_net, 0.01, 0.15, 20)
legend = ['baseline','Degree','vulnerable','Alzheimer']
viz = DiffusionTrendComparison([mod_norm, mod_degree, mod_vul, mod_alzh], 
[trends_normal, trends_degree, trends_vul, trends_alzh])
viz.plot("trend_comparison.pdf")

'''
viz = DiffusionTrend(model, trends)
p = viz.plot(width=400, height=400)
multi.add_plot(p)

viz2 = DiffusionPrevalence(model, trends)
p2 = viz2.plot(width=400, height=400)
show (p)
#multi.add_plot(p2)
#show(multi.plot())
'''

# %% CONTINUOUS CODE ATTEMPT
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
    neigh_score = av_state/len(graph.neighbors[node])
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
