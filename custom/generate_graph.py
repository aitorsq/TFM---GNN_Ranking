import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
np.random.seed(1)


def create_graph(graph_type,min_nodes,max_nodes):

    num_nodes = np.random.randint(min_nodes,max_nodes)

    if graph_type == "ER":
        #Erdos-Renyi random graphs
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = True)
        return g_nx

    if graph_type == "SF":
        #Scalefree graphs
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        return g_nx


    if graph_type == "GRP":
        #Gaussian-Random Partition Graphs
        s = np.random.randint(200,1000)
        v = np.random.randint(200,1000)
        p_in = np.random.randint(2,25)*0.0001
        p_out = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = True)
        assert nx.is_directed(g_nx)==True,"Not directed"
        return g_nx


def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    return g_nkit

def cal_exact_bet(g_nkit):

    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict




min_nodes = 50000
max_nodes = 100000
num_of_graphs = 50
graph_types = ["ER","SF","GRP"]

for graph_type in graph_types:
    print("###################")
    print(f"Generating graph type : {graph_type}")
    print(f"Number of graphs to be generated:{num_of_graphs}")
    list_bet_data = list()
    print("Generating graphs and calculating centralities...")
    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
        g_nx = create_graph(graph_type,min_nodes,max_nodes)
        
        if nx.number_of_isolates(g_nx)>0:
            #print("Graph has isolates.")
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data.append([g_nx,bet_dict])

    fname_bet = f"./graphs/{graph_type}_{num_of_graphs}_{max_nodes}_{min_nodes}.pickle"    

    with open(fname_bet,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)

    print("")
    print("Graphs saved")

print("End.")


        


