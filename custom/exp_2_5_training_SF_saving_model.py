 
from utils import *
from model_bet import *
import argparse
import time



for size in [100000,300000,900000]:
    for nodes in [10,100,1000,10000]:
        for c in [1,10,20,40]:
        
            torch.manual_seed(15)
            print(f"Size: {size}, Nodes: {nodes}, copies:{c}")

            data_path = 'SF_5_graphs_'+f"{c}_copies_{nodes}_nodes_{size}_size.pickle"
            
            #Load training data
            print(f"Loading data...")
            with open("./data_splits/train/"+data_path,"rb") as fopen:
                list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

            #Get adjacency matrices from graphs
            print(f"Graphs to adjacency conversion.")

            list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,size)


            #Model parameters
            hidden = 20

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
            num_epoch = 15

            print("Training")
            print(f"Total Number of epoches: {num_epoch}")
            
            for e in range(num_epoch):
                
                print(time.strftime("%H:%M:%S"))
                print(f"Epoch number: {e+1}/{num_epoch}")
                train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model=model,device=device,optimizer=optimizer,size=size)

                saving_model_path = f'./models/size_{size}/SF_5_graphs_'+f'{c}_copies_{nodes}_nodes_{size}_size_{e}_epochs'

                torch.save(model.state_dict(), saving_model_path)


