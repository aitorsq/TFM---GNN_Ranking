from utils import *
random.seed(1)


param = {
    "size" : [10000,100000,300000,900000],
    "num_train" : 5,
    "num_test" : 10,
    "num_copies": [1,10,20,40]
}


with open(f"./graphs/LFR_15_graphs.pickle","rb") as fopen:
    list_data = pickle.load(fopen)

num_graph = len(list_data)
assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."

for size in param["size"]:
    for c in param["num_copies"]:

        #For training split
        if param["num_train"] > 0:
            list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = c,adj_size=size)

            with open(f"./data_splits/train/LFR_5_graphs_{c}_copies_{size}_size.pickle","wb") as fopen:
                pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)

#For test split
size = param["size"][0]
if param["num_test"] > 0:
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[param["num_train"]:param["num_train"]+param["num_test"]],num_copies = 1,adj_size=size)

    with open(f"./data_splits/test/LFR_10_graphs_{size}_size.pickle","wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)