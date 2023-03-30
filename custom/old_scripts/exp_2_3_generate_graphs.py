from utils import *
import time
random.seed(1)

param = {
    "min_nodes": 5000,
    "max_nodes": 5001,
    "num_of_graphs": 5,
    "graph_types": ["SF"],
    "logfile":"../logfile.txt"
}

for nodes in [10,100,1000,10000,100000]:
    for graph_type in param["graph_types"]:
        print("###################")
        print(f"Generating graph: {nodes} nodes")
        print(f"Number of graphs to be generated:{param['num_of_graphs']}")
        list_bet_data = list()
        print("Generating graphs and calculating centralities...")
        for i in range(param['num_of_graphs']):
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Graph index:{i+1}/{param['num_of_graphs']}")
            g_nx = create_graph(graph_type,nodes,nodes+1)

            if nx.number_of_isolates(g_nx)>0:
                #print("Graph has isolates.")
                g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
                g_nx = nx.convert_node_labels_to_integers(g_nx)
            g_nkit = nx2nkit(g_nx)
            bet_dict = cal_exact_bet(g_nkit)
            list_bet_data.append([g_nx,bet_dict])

        fname_bet = f"./graphs/{graph_type}_{param['num_of_graphs']}_graphs_{nodes}_nodes.pickle"    

        with open(fname_bet,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)

        print("")
        print("Graphs saved")