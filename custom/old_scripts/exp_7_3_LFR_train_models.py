 
from utils import *
from model_bet import *
import argparse
torch.manual_seed(15)


param = {
    "size" : [10000,100000,300000,900000],
    "num_copies": [1,10,20,40]
}

#data_test = f"LFR_10_graphs_10000_size.pickle"
##Load test data
#with open("./data_splits/test/"+data_test,"rb") as fopen:
#    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)
#
#list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,size)

for size in param["size"]:
    for c in param["num_copies"]:

        torch.manual_seed(15)
        data_train = f"LFR_5_graphs_{c}_copies_{size}_size.pickle"    

        #Load training data
        print(f"Loading data...")
        with open("./data_splits/train/"+data_train,"rb") as fopen:
            list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

        list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,size)

        #Model parameters
        hidden = 20

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
        num_epoch = 15

        for e in range(num_epoch):
            print(f"{c}_copies_{size}_size_{e}_epoch_{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)
            
            saving_path = f"./models/LFR/LFR_5_graphs_{c}_copies_{size}_size_{e}_epoch"
            torch.save(model.state_dict(), saving_path)