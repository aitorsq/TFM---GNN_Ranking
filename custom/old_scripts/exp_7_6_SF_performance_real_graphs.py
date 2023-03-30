from utils import *
from model_bet import *
import argparse
import pandas as pd


graphs = ['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']
sizes = [10000,100000,300000,900000]
epochs = 5

Results = {}

for i in range(len(graphs)):

    g = graphs[i]
    size = sizes[i]
    data_test = f'{g}_{size}_size.pickle'

    Results[data_test] = {'true': [],'pred': []}
    #Load test data
    with open("./data_splits/test/"+data_test,"rb") as fopen:
        list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)
    list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,size)

    for c in [1,10,20,40]:        
        
        data_train = f"SF_5_graphs_10000_nodes_{c}_copies_{size}_size.pickle"
        
        #Load training data
        with open("./data_splits/train/"+data_train,"rb") as fopen:
            list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

        #Get adjacency matrices from graphs
        list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,size)

        for seed in range(10):
            
            currentresult = {'data_train': data_train, 'seed':seed, 'copies': c}
            
            torch.manual_seed(seed)
            print(f"G:{g}, size: {size}, copies: {c}, seed {seed}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            
            hidden = 20
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

#            with torch.no_grad():
#                r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)
#
#            currentresult['no_train'] = {'pred':r['pred'],'kt':r["kt"]}

            train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model=model,device=device,optimizer=optimizer,size=size)

            with torch.no_grad():
                r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)

            currentresult['train'] = {'pred':r['pred'],'kt':r["kt"]}

            if len(Results[data_test]['true']) == 0:
                Results[data_test]['true'] = r['true']

            Results[data_test]['pred'].append(currentresult)

            with open(f"SF_real_performance_{epochs}_epochs.pickle","wb") as fopen:
                pickle.dump(Results,fopen)
