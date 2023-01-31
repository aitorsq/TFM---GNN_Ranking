from utils import *
random.seed(1)

param = {
    "min_nodes": 5000,
    "max_nodes": 10000,
    "num_of_graphs": 15,
    "graph_types": ["ER","SF","GRP"],
    "logfile":"../logfile.txt"
}

log("###############################################################################################",param["logfile"])
log("Start generating graphs...",param["logfile"])
for graph_type in param["graph_types"]:
    print("###################")
    log(f"Generating graph type : {graph_type}",param["logfile"])
    print(f"Generating graph type : {graph_type}")
    print(f"Number of graphs to be generated:{param['num_of_graphs']}")
    list_bet_data = list()
    print("Generating graphs and calculating centralities...")
    for i in range(param['num_of_graphs']):
        log(f"Graph index:{i+1}/{param['num_of_graphs']}",param["logfile"])
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: Graph index:{i+1}/{param['num_of_graphs']}")
        g_nx = create_graph(graph_type,param['min_nodes'],param['max_nodes'])
        
        if nx.number_of_isolates(g_nx)>0:
            #print("Graph has isolates.")
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data.append([g_nx,bet_dict])

    fname_bet = f"./graphs/{graph_type}_{param['num_of_graphs']}graphs_{param['max_nodes']}_{param['min_nodes']}_nodes.pickle"    

    with open(fname_bet,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)

    print("")
    print("Graphs saved")

log("Finished",param["logfile"])
log("###############################################################################################",param["logfile"])