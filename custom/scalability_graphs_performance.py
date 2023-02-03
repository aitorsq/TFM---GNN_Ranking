import time
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt

nodes = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
edges = [2,4,6]

results = []

for n in nodes:
    for e in edges:
        print(f"Processing graph {n} nodes and {e} edges.")
        file = f'ER_1_graph_{n}_nodes_{e}_edges.pickle'
        #Load data
        with open(f"./data_splits_scalability_test/{file}","rb") as fopen:
            list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

        model_size = bc_mat_test.shape[0]

        #Get adjacency matrices from graphs
        print(f"Graphs to adjacency conversion.")
        starting = time.time()
        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)
        end = time.time()

        results.append({"n": n, "e": e, "t": round(end-starting,4)})

with open(f"auxresults.pickle","wb") as fopen:
            pickle.dump(results,fopen)

for e in edges:
    xs = []
    ys = []
    if e == 2:
        c = 'r'
        l = 'ratio 2'
    elif e == 4:
        c = 'b'
        l = 'ratio 4'
    else:
        c = 'g'
        l = 'ratio 6'
    for j in results:
        if j["e"] == e:
            xs.append(j["n"])
            ys.append(j["t"])
            plt.plot(xs,ys,color=c,legend=l)
            plt.scatter(xs,ys,color=c)
plt.legend()
plt.show()