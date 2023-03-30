
from utils import *
random.seed(1)


param = {
    "logfile": "../logfile.txt",
    "num_train" : 0,
    "num_test" : 1,
    "num_copies": 1,
    "files" : ['4-web-Google']
}

#['1-wiki-Vote','2-soc-Epinions','3-email-EuAll','4-web-Google']


for file in param["files"]:
    for size in [900000]:
    
        print(f"File: {file}, Size: {size}")
        with open(f"./real_graphs/bet_real_graphs/{file}.pickle","rb") as fopen:
            list_data = pickle.load(fopen)

        num_graph = len(list_data)
        assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."
        

        #For test split
        if param["num_test"] > 0:
            list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[param["num_train"]:param["num_train"]+param["num_test"]],num_copies = 1,adj_size=size)

            with open(f"./data_splits/test/{file}_{size}_size.pickle","wb") as fopen:
                pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
