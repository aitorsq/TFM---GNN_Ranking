
from utils import *
import os
random.seed(1)


param = {
    "graph_folder" : "graphs_scalability_test",
}

nodes = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
edges = [2,4,6]

for n in nodes:
    for e in edges:
        file = f'ER_1_graph_{n}_nodes_{e}_edges.pickle'
        with open(f"./{param['graph_folder']}/{file}","rb") as fopen:
            list_data = pickle.load(fopen)
        assert len(list_data) == 1
        num_graph = len(list_data)

        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data,num_copies = 1,adj_size=n)

        with open(f"./data_splits_scalability_test/{file}","wb") as fopen:
            pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen)
