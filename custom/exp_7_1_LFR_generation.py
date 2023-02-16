from networkit import *
import pickle
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from datetime import datetime

def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    assert g_nx.number_of_nodes()==g_nkit.numberOfNodes(),"Number of nodes not matching"
    assert g_nx.number_of_edges()==g_nkit.numberOfEdges(),"Number of edges not matching"
        
    return g_nkit

def cal_exact_bet(g_nkit):

    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict


def generate_bet_LFR_data(num_of_graphs,output_path):
    
    list_bet_data = list()

    for i in range(num_of_graphs):
        
        while True:
            try:
                print(f"Graph index:{i+1}/{num_of_graphs}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                g_nx = LFR_benchmark_graph(n=10000,tau1=3,tau2=1.5,mu=0.05,average_degree=6,min_community=20)
            except:
                continue
            else:
                break
        print("removing isolates")
        
        if nx.number_of_isolates(g_nx)>0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
        
        g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data.append([g_nx,bet_dict])

        with open(f"custom/"+output_path,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)


# Create train graphs and save them to sf_train_50.pickle
num_of_graphs = 15

output_path = f"graphs/LFR_{num_of_graphs}_graphs.pickle"

generate_bet_LFR_data(num_of_graphs,output_path)