from utils import *
from model_bet import *
import argparse
import pandas as pd


graphs = ['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']
sizes = [10000,100000,300000,900000]

graphs = ['1-wiki-Vote']
sizes = [10000]

Results = {'LFR': {}, 'SF': {}}

for i in range(len(graphs)):

    g = graphs[i]
    size = sizes[i]
    data_test = f'{g}_{size}_size.pickle'

    #Load test data
    with open("./data_splits/test/"+data_test,"rb") as fopen:
        list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

    list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,size)

    Results['LFR'][f"{size}_size"] = {'test_graph': data_test, 'real': []}
    Results['SF'][f"{size}_size"] = {'test_graph': data_test, 'real': []}

    for c in [1,10,20,40]:
    
        LFR_data_train = f"LFR_5_graphs_{c}_copies_{size}_size.pickle"
        SF_data_train = f"SF_5_graphs_10000_nodes_{c}_copies_{size}_size.pickle"
        
        Results['LFR'][f"{size}_size"][f"{c}_copies"] = {'data_train' : LFR_data_train,'pred':{}}
        Results['SF'][f"{size}_size"][f"{c}_copies"] = {'data_train' : SF_data_train,'pred':{}}
        
        for epoch in range(15):
            
            Results['LFR'][f"{size}_size"][f"{c}_copies"]['pred'][f'{epoch}_epoch'] = {}
            Results['SF'][f"{size}_size"][f"{c}_copies"]['pred'][f'{epoch}_epoch'] = {}
        
            for seed in range(15):
                    
                    data_train = LFR_data_train
                    model_path = f'{data_train[:-7]}_{seed}_seed_{epoch}_epoch'
                    print(model_path)
                    torch.manual_seed(seed)
                    hidden = 20
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
                    model.load_state_dict(torch.load(f'models/{model_path}'))
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
                    with torch.no_grad():
                        r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)

                    Results['LFR'][f"{size}_size"][f"{c}_copies"]['pred'][f'{epoch}_epoch'][f"{seed}_seed"] = {'pred':r['pred'],'kt':r["kt"]}

                    if len(Results['LFR'][f"{size}_size"]['real']) == 0:
                        Results['LFR'][f"{size}_size"]['real'] = r['true']

                    with open(f"LFR_real_performance_full_{g}.pickle","wb") as fopen:
                        pickle.dump(Results,fopen)

                    data_train = SF_data_train
                    model_path = f'{data_train[:-7]}_{seed}_seed_{epoch}_epoch'
                    torch.manual_seed(seed)
                    hidden = 20
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
                    model.load_state_dict(torch.load(f'models/{model_path}'))
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
                    with torch.no_grad():
                        r = test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,model=model,device=device,size=size)

                    Results['SF'][f"{size}_size"][f"{c}_copies"]['pred'][f'{epoch}_epoch'][f"{seed}_seed"] = {'pred':r['pred'],'kt':r["kt"]}

                    if Results['SF'][f"{size}_size"]['real'] == -1:
                        Results['SF'][f"{size}_size"]['real'] = r['true']

                    with open(f"SF_real_performance_full_{g}.pickle","wb") as fopen:
                        pickle.dump(Results,fopen)
