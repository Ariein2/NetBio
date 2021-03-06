# %% IMPORT PACKAGES: 
# The packages listed must be imported following the instructions 
# on their respective documentation pages. It is also necessary to 
# install the extra dependencies they mention, specially for the 
# NDLIB package. Moreover, for the NDlib package it is necessary to 
# install the development version through their github following the
# instructions described in the documentation . 

# NOTE: The code contains random selections so every iteration can lead to different 
# results. 

import networkx as nx
import numpy as np 
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import matplotlib.pyplot as plt
from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence
import matplotlib as plot
import selenium
import hvplot.networkx as hvnx
from itertools import compress
import pandas as pd
import random
from ndlib.viz.mpl.TrendComparison import DiffusionTrendComparison

#%% PREPROCESSING: 
# Load graphs and remove isolated nodes to only study the main 
# connected component.

# LOAD GRAPH FOR MAIN NETWORK (SCENARIO 1)
raw_net = nx.read_graphml('budapest_large.graphml')
raw = hvnx.draw(raw_net, node_color = 'lightblue')
print('Properties raw_net: '+ str(len(raw_net.nodes())) 
      + ' and ' + str(len(raw_net.edges())))

# Remove isolated nodes
main_net = raw_net 
main_net.remove_nodes_from(list(nx.isolates(main_net)))
processed = hvnx.draw(main_net, node_color = 'lightblue')
print('Properties main_net: '+ str(len(main_net.nodes())) 
      +' and '+ str(len(main_net.edges())))

# LOAD NETWORK SCENARIO 2
deg_net = nx.read_graphml('budapest_large.graphml')
deg_net.remove_nodes_from(list(nx.isolates(deg_net)))

# LOAD NETWORK SCENARIO 3 
vul_net = nx.read_graphml('budapest_large.graphml')
vul_net.remove_nodes_from(list(nx.isolates(vul_net)))

# LOAD NETWORK SCENARIO 4 
alzh_net = nx.read_graphml('budapest_large.graphml')
alzh_net.remove_nodes_from(list(nx.isolates(alzh_net)))

# Visualise networks 
layout = raw + processed
layout

#%% SCENARIO 2: Isolate high degree nodes 
# Isolate 10 high degree nodes. High degree nodes are selected
# randomly from the set of nodes in the top quartile of the
# node degree distribution. 

# Identify high degree nodes

degree = list(dict(deg_net.degree).values())

# Identify top quartile degree nodes: 
quartile = np.quantile(degree, 0.75) 
keys = list(dict(deg_net.degree).keys())
bool_quart = degree >= quartile
quartile_keys = list(compress(keys, bool_quart))

# Select a random set of top quartile nodes to isolate
nodes_removed = 10
selected_nodes = random.sample(quartile_keys,nodes_removed)
edge_remove = list(deg_net.edges(selected_nodes))
deg_net.remove_edges_from(edge_remove)

# Draw deg_net 
print('Properties deg_net: '+ str(len(deg_net.nodes())) 
      +' and '+ str(len(deg_net.edges())))
hvnx.draw(deg_net, node_color = 'lightblue')

# %% SCENARIO 3: Isolate vulnerable nodes 
# Isolate 10 vulnerable nodes in the network. Vulnerable nodes are 
# nodes which have an influence 1/degree which is smaller than the threshold.
# These nodes are the most relevant from the dynamical point of view. 
 
# Identify influence in nodes (influence = 1/degree)
degree = list(dict(vul_net.degree).values())
influence = []
for deg in degree: 
    if (deg == 0):
        influence.append(0)
    else:
        influence.append(1/deg)

# Identify vulnerable nodes
threshold = np.float64(0.15)
bool_inf = influence >= threshold

# Select random subset of vulnerable nodes 
nodes_removed = 10
keys = list(dict(vul_net.degree).keys())
vulnerable_keys = list(compress(keys, bool_inf))
selected_nodes = random.sample(vulnerable_keys, nodes_removed)
edge_remove = list(vul_net.edges(selected_nodes))
vul_net.remove_edges_from(edge_remove)

# Draw vul_net
print('Properties vul_net: '+ str(len(vul_net.nodes())) 
       +' and '+ str(len(vul_net.edges()))) 
hvnx.draw(vul_net, node_color = 'lightblue')


#%% SCENARIO 4: Alzheimer related area
# Isolate 10 nodes which are located in regions related to Alzheimer's 
# disease. 

#Identify nodes related to areas of interest 
node_info = pd.DataFrame(alzh_net.nodes._nodes)
areas = node_info.loc[['dn_fsname']]

area_bool = []
for index, text in areas.iteritems() :
    string = str(text)
    result = string.find ('Hippocampus')
    result2 = string.find ('middletemporal')
    if (result == -1 & result2 == -1):
        area_bool.append(False)
    else: 
        area_bool.append(True)

keys = list(dict(alzh_net.degree).keys())
alzh_keys = list(compress(keys, area_bool))

# Select a random subset of nodes in the area
nodes_removed = 10

selected_nodes = random.sample(alzh_keys, nodes_removed)
edge_remove = list(alzh_net.edges(selected_nodes))
alzh_net.remove_edges_from(edge_remove)

#Draw alzh_net
print('Properties alzh_net: '+ str(len(alzh_net.nodes())) 
      +' and '+ str(len(alzh_net.edges())))
hvnx.draw(alzh_net, node_color = 'lightblue')

#%% MODEL AND SIMULATION DEFINITION 
# Implement threshold model with a threshold parameter of 
# 0.15 a starting fraction of infected nodes of 0.01 and 

#Function definition 
def model (net, frac_inf, threshold, iter):
    # Defines the model type, specification and implementation. 
    # The function uses the network input in net as the main input
    # for the threshold model. Then the rest of parameters are 
    # specified in the model configuration: 
    # frac_inf: Initial fraction of infected nodes. 
    # threshold: Fraction of active neighbours necessary to change 
    # a node's state to active. 
    # iter: Number of iterations to run from the dynamical process. 

    # Model impolementation
    model = ep.ThresholdModel(net)

    # Model Configuration: initial conditions
    config = mc.Configuration()
    config.add_model_parameter('fraction_infected', frac_inf)

    for i in net.nodes():
        config.add_node_configuration("threshold", i, threshold)

    model.set_initial_status(config)

    # Simulation execution
    iterations = model.iteration_bunch(iter)
    trends = model.build_trends(iterations)

    return [model, iterations, trends]

#Parameter definition
timesteps = 20
threshold = 0.15
frac_inf = 0.01
# Run models in the different scenarios
[mod_norm, iter_norm, trends_normal] = model(main_net, frac_inf, threshold, timesteps)
[mod_degree, iter_deg, trends_degree] = model(deg_net, frac_inf, threshold, timesteps)
[mod_vul, iter_vul, trends_vul] = model(vul_net, frac_inf, threshold, timesteps)
[mod_alzh, iter_alzh, trends_alzh] = model(alzh_net, frac_inf, threshold, timesteps)

#%% VISUALIZATION:
# Prepare networks for visualisation and create overall plot. 
# Save states on the simulation as variables in the network and also 
# save the se

#Function definition
def state_complete (iter_norm):
    # Creates a dictionary containing the state of all nodes 
    # at each timestep. Uses as input the iterations otuput from 
    # the model () function. 

    iter_complete = {}
    for it in iter_norm: 
        if (it['iteration'] == 0):
            iter_complete[it['iteration']] = iter_norm[0]['status']
        else :
            iter_complete[it['iteration']] = {**iter_complete[it['iteration']-1], 
                                              **iter_norm[it['iteration']]['status']}
    return iter_complete


def save_status (graph, iteration):
    # Stores as a graph variable the state (active=1 inactive=0) 
    # of all nodes at 3 different iterations.
    # At the start (stat0), half way through the dynamics (stat10) 
    # and end (stat19).

    for node in graph.nodes:
        #save state in grpah 
        graph.nodes[node]['stat0'] = iteration[0][node]
        graph.nodes[node]['stat10'] = iteration[10][node]
        graph.nodes[node]['stat19'] = iteration[19][node]
    return graph

def save_variables (graph, boolean, scenario):
    # Stores as a graph variable the information regarding the nodes 
    # of interest at each scenario. 
    # Scenario 2: Assigns 1 to nodes in the top quartile, 
    # assigns 0 otherwise.
    # Scenario 3: Assigns 1 to nodes with 1/degree <=threshold 
    # (vulnerable nodes), assigns 0 otherwise.
    # Scenario 4: Assigns 1 to nodes in the areas related to 
    # Alzheimer's disease, assigns 0 otherwise.

    for i, node in enumerate(graph.nodes):
        bin_bool = []

        if (scenario == 2):
            bin_bool = boolean*1
            graph.nodes[node]['high_deg'] = list(bin_bool)[i].item()

        elif (scenario == 3):
            bin_bool = boolean*1
            graph.nodes[node]['vulnerable'] = list(bin_bool)[i].item()
        
        elif (scenario == 4):
            bin_bool = np.array(boolean)*1
            graph.nodes[node]['Alzh'] = list(bin_bool)[i].item()
    
    return graph

# Create complete list of states: 
complete_norm = state_complete (iter_norm)
complete_deg = state_complete (iter_deg)
complete_vul = state_complete (iter_vul)
complete_alzh = state_complete (iter_alzh)

av_norm = []
av_deg = []
av_vul = []
av_alzh = []

# Proportion of active nodes at each timempoint: 
for di in complete_norm:
    av_norm.append(np.average(np.array(list(complete_norm[di].values()))))
    av_deg.append(np.average(np.array(list(complete_deg[di].values()))))
    av_vul.append( np.average(np.array(list(complete_vul[di].values()))))
    av_alzh.append(np.average(np.array(list(complete_alzh[di].values()))))

#Plot % of active nodes 
plt.figure ()
plt.title('Propagation progression')
plt.plot(av_norm)
plt.plot(av_deg)
plt.plot(av_vul)
plt.plot(av_alzh)
plt.xlabel('Iteration')
plt.xlim([0,19])
plt.ylabel('Proportion of active nodes')
plt.legend(['Baseline network','Hub nodes','Vulnerable nodes','Alzheimer scenario'])
plt.savefig('proportionplot.svg')
# Add information to graphs and export them for visualization

graph_normal = save_status (main_net, complete_norm)
graph_normal = save_variables (graph_normal, area_bool, 4)
graph_normal = save_variables (graph_normal, bool_inf, 3)
graph_normal = save_variables (graph_normal, bool_quart, 2)
nx.write_graphml(graph_normal,'sim_baseline.graphml')

graph_deg = save_status (deg_net, complete_deg)
nx.write_graphml(graph_deg,'sim_high_degree.graphml')

graph_vul = save_status (vul_net, complete_vul)
nx.write_graphml(graph_vul,'sim_vulnerable.graphml')

graph_alzh = save_status (alzh_net, complete_alzh)
nx.write_graphml(graph_alzh,'sim_alzheimer.graphml')

#%%