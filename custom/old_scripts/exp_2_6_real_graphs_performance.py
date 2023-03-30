
 
from utils import *
from model_bet import *
import argparse
import pandas as pd
torch.manual_seed(15)

graphs = ['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']
sizes = [10000,100000,300000,900000]

Results = { "graph":[],
            "nodes": [],
            "size":[],
            "copies":[],
            "epochs": [],
            "kendalltau":[],
            "avg":[]}

for g in graphs:
    for size in sizes[int(g[0])-1:]:
        data_path = f'{g}_{size}_size.pickle'
        print(f"G:{g}, size: {size}")

        #Load test data
        with open("./data_splits/test/"+data_path,"rb") as fopen:
            list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

        #Get adjacency matrices from graphs
        #print(f"Graphs to adjacency conversion.")

        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,size)

        for nodes in [10,100,1000,10000]:
            for c in [1,10,20,40]:
                for e in range(15):
                    print(f"G:{g}, size: {size}, nodes: {nodes}, copies: {c}, epoch {e}")
                    model_path = f"./models/size_{size}/SF_5_graphs_{c}_copies_{nodes}_nodes_{size}_size_{e}_epochs"

                    #Model parameters
                    hidden = 20

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)

                    model.load_state_dict(torch.load(model_path))

                    model.to(device)


                    with torch.no_grad():
                        r = test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)
                    
                    Results["graph"].append(g)
                    Results["nodes"].append(nodes)
                    Results["size"].append(size)
                    Results["copies"].append(c)
                    Results["epochs"].append(e)
                    Results["kendalltau"].append(r["kt"])
                    Results["avg"].append(r["avg"])


                    df = pd.DataFrame.from_dict(Results)
                    df.to_csv("output_real_graphs_peformance_full.csv")