
from utils import *
from model_bet import *
import pandas as pd
torch.manual_seed(15)
import time

test_files = {
    10000: '1-wiki-Vote_10000_size.pickle',
    100000: '2-soc-Epinions_100000_size.pickle',
    300000: '3-email-EuAll_300000_size.pickle',
    900000: '4-web-Google_900000_size.pickle'}

copies = 10
nodes = 100

R = {}

for size in test_files:

    test_graph = test_files[size]
    train_graph = f'SF_5_graphs_{copies}_copies_{nodes}_nodes_{size}_size.pickle'
    R[test_graph] = {"train_graph":train_graph,"size":size}
    R[test_graph]['r'] = {'true': [],'pred': []}

    #Load test data
    with open("./data_splits/test/"+test_graph,"rb") as fopen:
        list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

    # Load training data
    with open("./data_splits/train/"+train_graph,"rb") as fopen:
        list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)


    #Get adjacency matrices from graphs
    list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,size)
    list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,size)

    for seed in range(50):

        currentresult = {'seed':seed}

        print(f"\nTest: {test_graph}, Size: {size}, Seed: {seed}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        model_path = f"./models/seeds_analysis/notrained/notrained_{size}_size_{seed}_seed"

        #Model parameters
        hidden = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

        with torch.no_grad():
            r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)

        currentresult['no_train'] = {'pred':r['pred'],'kt':r["kt"]}
        print(f'no trained: {r["kt"]}')

        if len(R[test_graph]['r']['true']) == 0:
            R[test_graph]['r']['true'] = r['true']

        train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model=model,device=device,optimizer=optimizer,size=size)

        with torch.no_grad():
            r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)
        
        print(f'trained: {r["kt"]}')

        currentresult['train'] = {'pred':r['pred'],'kt':r["kt"]}

        R[test_graph]['r']['pred'].append(currentresult)

        with open("train_no_train_data.pickle","wb") as fopen:
            pickle.dump(R,fopen)