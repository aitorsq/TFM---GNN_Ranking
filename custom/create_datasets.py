
from utils import *
random.seed(1)


param = {
    "logfile": "../logfile.txt",
    "adj_size" : 10000,
    "graph_types" : ["ER","SF","GRP"],
    "num_train" : 5,
    "num_test" : 10,
    "num_copies": 6,
    "files" : ["ER_15_graphs_10000_5000_nodes.pickle",
               "SF_15_graphs_10000_5000_nodes.pickle",
               "GRP_15_graphs_10000_5000_nodes.pickle"]
}


log("###############################################################################################",param["logfile"])
log("Loading graphs from pickle files for splitting data...",param["logfile"])
print("Loading graphs from pickle files for splitting data...")
log("Adj size: 10000, num_train: 5, num_test: 10, num_copies: 6",param["logfile"])

for file in param["files"]:
    log(f"Spliting {file}",param["logfile"])
    with open(f"./graphs/{file}","rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    if param["num_train"] > 0:
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = param["num_copies"],adj_size=param["adj_size"])

        with open(f"./data_splits/train/{file}","wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
    if param["num_test"] > 0:
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[param["num_train"]:param["num_train"]+param["num_test"]],num_copies = 1,adj_size=param["adj_size"])

        with open(f"./data_splits/test/{file}","wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

log(f"Finished",param["logfile"])
print(" Data split saved.")
log("###############################################################################################",param["logfile"])
