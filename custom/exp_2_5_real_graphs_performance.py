 
from utils import *
from model_bet import *
import argparse
torch.manual_seed(15)

model_size = 10000
data_path = '1-wiki-Vote_10000_size.pickle'

#Load test data
with open("./data_splits/test/"+data_path,"rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.")

list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)


def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
    
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")


model_path = f"./models/trained_{model_size}_size_5_SF_10000_nodes"

#Model parameters
hidden = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Bet(ninput=model_size,nhid=hidden,dropout=0.6)

model.load_state_dict(torch.load(model_path))

model.to(device)


with torch.no_grad():
    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)