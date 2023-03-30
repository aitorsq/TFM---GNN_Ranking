from utils import *

def read_compute_target(in_path,out_path):
    
    #print("Computing Bet")
    list_bet_data = list()
    g_nx = nx.read_edgelist(in_path, comments='#', nodetype=int)
    if nx.number_of_isolates(g_nx)>0:
        #print("Graph has isolates.")
        g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
    g_nx = nx.convert_node_labels_to_integers(g_nx)
    g_nkit = nx2nkit(g_nx)
    bet_dict = cal_exact_bet(g_nkit)
    list_bet_data.append([g_nx,bet_dict])
    
    with open(out_path,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)
    
    print(f"Bet computed and graph saved: {out_path}")


networks = ['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']

for net in networks:
    in_path_graph = f"./real_graphs/original/{net}.txt"
    out_path_graph = f"./real_graphs/bet_real_graphs/{net}.pickle"
    read_compute_target(in_path_graph,out_path_graph)