from utils import *
from model_bet import *
import argparse
import pandas as pd
import multiprocessing as mp

def paral_func(size,copies,seed,list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    
    print(f"Starting: size:{size}, copies: {copies},seed: {seed}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")    
    torch.manual_seed(seed)           
    hidden = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN_Bet(ninput=size,nhid=hidden,dropout=0.6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

    epochs = 15
    for e in range(epochs):
        train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model=model,device=device,optimizer=optimizer,size=size)
    
        saving_path = f'models/LFR/LFR_5_graphs_{copies}_copies_{size}_size_{seed}_seed_{e}_epoch'
        torch.save(model.state_dict(), saving_path)
    
        print(f"Finished {saving_path}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")


if __name__ == "__main__":

    graphs = ['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']
    sizes = [10000,100000,300000,900000]

    for i in range(len(graphs)):

        size = sizes[i]

        #for c in [1,10,20,40]:
        for c in [10]:
        
            data_train = f"LFR_5_graphs_{c}_copies_{size}_size.pickle"
            
            #Load training data
            with open("./data_splits/train/"+data_train,"rb") as fopen:
                list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)

            #Get adjacency matrices from graphs
            list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,size)

            print(f"Starting: Size: {size}, copies: {c}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            
            processes = []
            
            seeds = 12

            for batch in range(seeds//4):
                for id in range(4):
                    seed = id+batch*4
                    #paral_func(size,c,seed,list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)
                    
                    p = mp.Process(target=paral_func,args=[size,c,seed,list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train])
                    p.start()
                    processes.append(p)


                for process in processes:
                    process.join()

            print(f"Finished: Size: {size}, copies: {c}, Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")