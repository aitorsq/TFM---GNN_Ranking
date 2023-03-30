 
from utils import *
from model_bet_varying_layers import *
import argparse
torch.manual_seed(15)
import pandas as pd

results = {'g': [], 'layers':[], 'epoch':[], 'kt':[], 'avg':[]}

for gtype in ['SF','ER','GRP']:
    for num_layers in range(1,8):

        torch.manual_seed(15)
        print(gtype)


        #data_path = gtype+"_15_graphs_10000_5000_nodes.pickle"
        data_path = gtype+"_train_given.pickle"
        #Load training data
        print(f"Loading data...")
        with open("./data_splits/train/"+data_path,"rb") as fopen:
            list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

        data_path = gtype+"_test_given.pickle"
        #Load test data
        with open("./data_splits/test/"+data_path,"rb") as fopen:
            list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

        model_size = bc_mat_train.shape[0]

        #Get adjacency matrices from graphs
        print(f"Graphs to adjacency conversion.")

        list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

        #Model parameters
        hidden = 20

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"GNN_Bet{num_layers}")
        exec(f'model = GNN_Bet{num_layers}(ninput=model_size,nhid=hidden,dropout=0.6)')
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        num_epoch = 15

        print("Training")
        print(f"Total Number of epoches: {num_epoch}")
        for e in range(num_epoch):
            print(f"Epoch number: {e+1}/{num_epoch}")
            train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model=model,device=device,optimizer=optimizer,size=model_size)

            #to check test loss while training
            with torch.no_grad():
                r = test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=model_size)
            
                results['g'].append(gtype)
                results['layers'].append(num_layers)
                results['epoch'].append(e)
                results['kt'].append(r["kt"])
                results['avg'].append(r["avg"])

df = pd.DataFrame.from_dict(results)
df.to_csv("output_varying_layers_given.csv")