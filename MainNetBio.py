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
import matplotlib.pyplot as plt
import hvplot.networkx as hvnx
import selenium
from itertools import compress 
import pandas as pd
import random 
from ndlib.utils import multi_runs
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
main_net = data 
shell = hvnx.draw(data, node_color = 'lightblue')
main_net.remove_nodes_from(list(nx.isolates(main_net)))
len(data.nodes())
'''
for node in data.nodes:
    if (data.nodes[node]["dn_hemisphere"] == "left"):
        data.nodes[node]["color"] = "ligthblue"
    else:
        data.nodes[node]["color"] = "darksalmon"
'''
shell2 = hvnx.draw(main_net, node_color = 'lightblue')
shell2 
'''
cy = CyRestClient()
network = cy.network.create_from_networkx(data)
'''
#layout =shell+shell2
#layout
#hvnx.save(layout,'layout.png')

#output_file(shell)
#%% HIGH DEGREE NODES 

# Find High degree nodes
deg_net = main_net
hvnx.draw(deg_net)
nodes_removed =10
degree = list(dict(deg_net.degree).values())

# Select nodes from top degree quartile: 
quartile = np.quantile(degree, 0.75) 
keys = list(dict(deg_net .degree).keys())
bool_quart = degree >= quartile
quartile_keys = list(compress(keys, bool_quart))
selected_nodes = random.sample(quartile_keys,nodes_removed)
edge_remove = list(deg_net.edges(selected_nodes))
deg_net.remove_edges_from(edge_remove)

#deg_net.remove_nodes_from(remove_nodes)
draw = hvnx.draw(deg_net, node_color = 'lightblue')
draw 
# %% VULNERABLE NODES 

# Find vulnerable nodes
vul_net = data
nodes_removed = 10
degree = list(dict(vul_net.degree).values())
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
keys = list(dict(vul_net.degree).keys())
vulnerable_keys = list(compress(keys, bool_inf))
selected_nodes = random.sample(vulnerable_keys, nodes_removed)
edge_remove = list(vul_net.edges(selected_nodes))
vul_net.remove_edges_from(edge_remove)

draw = hvnx.draw(vul_net, node_color = 'lightblue')
draw 
# nx.draw(data)

#%% Alzheimer related area
alzh_net = main_net

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
hvnx.draw(alzh_net, node_color = 'lightblue')

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
    #trends = multi_runs(model, execution_number=1, iteration_number=iter, 
    #                    infection_sets=None)
    return [model, iterations, trends]
timesteps = 20
threshold = 0.15

[mod_norm, iter_norm, trends_normal] = model(main_net, 0.01, 0.15, timesteps)
[mod_degree, iter_deg, trends_degree] = model(deg_net, 0.01, 0.15, timesteps)
[mod_vul, iter_vul, trends_vul] = model(vul_net, 0.01, 0.15, timesteps)
[mod_alzh, iter_alzh, trends_alzh] = model(alzh_net, 0.01, 0.15, timesteps)
viz = DiffusionTrendComparison([mod_norm, mod_degree, mod_vul, mod_alzh], 
[trends_normal, trends_degree, trends_vul, trends_alzh])
viz.plot("trend_comparison.pdf")
#plt.plot(trends_normal, trends_degree, trends_vul, trends_alzh)
half_time = np.round(timesteps/2)
#%%
def state_graph (iter_norm):
    iter_complete={}
    for it in iter_norm: 
        if (it['iteration'] == 0):
            iter_complete[it['iteration']] = iter_norm[0]['status']
        else :
            iter_complete[it['iteration']] = {**iter_complete[it['iteration']-1], 
                                                    **iter_norm[it['iteration']]['status']}
    return iter_complete

def save_status (graph, iteration):
    for node in graph.nodes:
        graph.nodes[node]['stat0'] = iteration[0][node]
        graph.nodes[node]['stat1'] = iteration[10][node]
        graph.nodes[node]['stat2'] = iteration[19][node]
    return graph

complete_norm = state_graph (iter_norm)
complete_deg = state_graph (iter_deg)
complete_vul = state_graph (iter_vul)
complete_alzh = state_graph (iter_alzh)

graph_normal = save_status (main_net, complete_norm)
nx.write_graphml(graph_normal,'normal_test.graphml')
graph_deg = save_status (deg_net, complete_deg)
graph_vul = save_status (vul_net, complete_vul)
graph_alzh = save_status (alzh_net, complete_alzh)

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
# %%
