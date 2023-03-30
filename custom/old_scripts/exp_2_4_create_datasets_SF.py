
from utils import *
random.seed(1)


param = {
    "logfile": "../logfile.txt",
    "num_train" : 5,
    "num_test" : 0,
}

for nodes in [10,100,1000,10000]:
    for size in [10000,100000,300000,900000]:
        for c in [1,10,20,40]:
        
            file = f"SF_5_graphs"
            print(f"Nodes {nodes}, size: {size}, copies: {c}")

            with open(f"graphs/{file}_{nodes}_nodes.pickle","rb") as fopen:
                list_data = pickle.load(fopen)

            num_graph = len(list_data)
            assert param["num_train"]+param["num_test"] == num_graph,"Required split size doesn't match number of graphs in pickle file."
            
            #For training split
            if param["num_train"] > 0:
                list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:param["num_train"]],num_copies = c, adj_size=size)

                with open(f"data_splits/train/{file}_{c}_copies_{nodes}_nodes_{size}_size.pickle","wb") as fopen:
                    pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)


print(" Data split saved.")
