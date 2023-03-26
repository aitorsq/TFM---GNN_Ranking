
from utils import *
random.seed(1)


param = {
    "adj_size" : 10000,
    "graph_types" : ["GRP","ER","SF"],
    "num_train" : 5,
    "num_test" : 10,
    "num_copies": [100],#[1,2,10,20,40],
    "files" : ["ER_15_graphs_10000_5000_nodes.pickle",
               "SF_15_graphs_10000_5000_nodes.pickle",
               "GRP_15_graphs_10000_5000_nodes.pickle"]
}


for file in param["files"]:
    for num_copies in param["num_copies"]:


        with open(f"./graphs/{file}","rb") as fopen:
            list_data = pickle.load(fopen)

        num_graph = len(list_data)
        assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
        #For training split
        if param["num_train"] > 0:
            list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = num_copies, adj_size=param["adj_size"])

            with open(f"./data_splits/train/{file[:-7]}_{num_copies}_copies_{param['num_train']}_train.pickle","wb") as fopen:
                pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

    #For test split
#    if param["num_test"] > 0:
#        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[param["num_train"]:param["num_train"]+param["num_test"]],num_copies = 1,adj_size=param["adj_size"])
#
#        with open(f"./data_splits/test/{file}","wb") as fopen:
#            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
