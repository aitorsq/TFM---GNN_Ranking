 
from utils import *
from model_bet import *
import pandas as pd

Results = { "gtype":[],
            "copies":[],
            "epochs": [],
            "kendalltau":[],
            "std":[]}

#for gtype in ["GRP","ER","SF"]:

for gtype in ["GRP"]:
        
    #Load test data
    #with open("./data_splits/test/"+f"{gtype}_15_graphs_10000_5000_nodes.pickle","rb") as fopen:
    with open("./data_splits/test.pickle","rb") as fopen:
        list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

#    for c in [1,2,10,20,40]:
    for c in [100]:
        print(f"{gtype}, {c} copies")
        
        data_path = f"{gtype}_15_graphs_10000_5000_nodes_{c}_copies_5_train.pickle"

        #Load training data
        print(f"Loading data...")
        #with open("./data_splits/train/"+data_path,"rb") as fopen:
        with open("./data_splits/training.pickle","rb") as fopen:
            list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

        model_size = bc_mat_train.shape[0]
        print(f"Model size: {model_size}")
        #Get adjacency matrices from graphs
        print(f"Graphs to adjacency conversion.")

        list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

        #Model parameters
        
        torch.manual_seed(15)

        hidden = 20

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        num_epoch = 15

        print("Training")
        print(f"Total Number of epoches: {num_epoch}")
        for e in range(num_epoch):
            print(f"Epoch number: {e+1}/{num_epoch}")
            train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model,device,optimizer,model_size)

            #to check test loss while training
            with torch.no_grad():
                r = test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model,device,model_size)

            Results["gtype"].append(gtype)
            Results["copies"].append(c)
            Results["epochs"].append(e)
            Results["kendalltau"].append(r["kt"])
            Results["std"].append(r["std"])


#            df = pd.DataFrame.from_dict(Results)
#            df.to_csv("output_synthetic_graphs_performance.csv")
