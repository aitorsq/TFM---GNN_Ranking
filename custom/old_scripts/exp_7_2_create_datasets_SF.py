from utils import *
random.seed(1)


param = {
    "size" : [10000,100000,300000,900000],
    "num_train" : 5,
    "num_test" : 0,
    "num_copies": [1,10,20,40]
}


with open(f"./graphs/SF_5_graphs_10000_nodes.pickle","rb") as fopen:
    list_data = pickle.load(fopen)

num_graph = len(list_data)
assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."

for size in param["size"]:
    for c in param["num_copies"]:

        #For training split
        if param["num_train"] > 0:
            list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = c,adj_size=size)

            with open(f"./data_splits/train/SF_5_graphs_10000_nodes_{c}_copies_{size}_size.pickle","wb") as fopen:
                pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
