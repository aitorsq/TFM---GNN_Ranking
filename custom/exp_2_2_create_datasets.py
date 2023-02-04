
from utils import *
random.seed(1)


param = {
    "logfile": "../logfile.txt",
    "adj_size" : 10000,
    "num_train" : 0,
    "num_test" : 1,
    "num_copies": 1,
    "files" : ["1-wiki-Vote"]
}


log("###############################################################################################",param["logfile"])
log("Loading graphs from pickle files for splitting data...",param["logfile"])
print("Loading graphs from pickle files for splitting data...")
log("Adj size: 10000, num_train: 5, num_test: 10, num_copies: 6",param["logfile"])

for file in param["files"]:
    log(f"Spliting {file}",param["logfile"])
    with open(f"./real_graphs/bet_real_graphs/{file}.pickle","rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    if param["num_train"] > 0:
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = param["num_copies"],adj_size=param["adj_size"])

        with open(f"./data_splits/train/{file}_{param['adj_size']}_size.pickle","wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    if param["num_test"] > 0:
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[param["num_train"]:param["num_train"]+param["num_test"]],num_copies = 1,adj_size=param["adj_size"])

        with open(f"./data_splits/test/{file}_{param['adj_size']}_size.pickle","wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

log(f"Finished",param["logfile"])
print(" Data split saved.")
log("###############################################################################################",param["logfile"])
