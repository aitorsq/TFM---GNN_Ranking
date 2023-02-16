import time
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt

nodes = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
edges = [2,4,6]

results = []

#for n in nodes:
#    for e in edges:
#        print(f"Processing graph {n} nodes and {e} edges.")
#        file = f'ER_1_graph_{n}_nodes_{e}_edges.pickle'
#        #Load data
#        with open(f"./data_splits_scalability_test/{file}","rb") as fopen:
#            list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)
#
#        model_size = bc_mat_test.shape[0]
#
#        #Get adjacency matrices from graphs
#        print(f"Graphs to adjacency conversion.")
#        starting = time.time()
#        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)
#        end = time.time()
#
#        results.append({"n": n, "e": e, "t": round(end-starting,4)})

#with open(f"auxresults.pickle","wb") as fopen:
#            pickle.dump(results,fopen)


with open(f"auxresults.pickle","rb") as fopen:
    results = pickle.load(fopen)


line_2 = []
line_4 = []
line_6 = []
xs = []

for j in results:
    if j["e"] == 2:
        line_2.append([j['n'],j['t']])
    elif j["e"] == 4:
        line_4.append([j['n'],j['t']])
    else:
        line_6.append([j['n'],j['t']])

print(line_2)
print(line_4)
print(line_6)


xs = [j[0]/100000 for j in line_2]
ys = [j[1] for j in line_2]
plt.plot(xs,ys,color='r',label='Ratio_2')
plt.scatter(xs,ys,color='r')

xs = [j[0]/100000 for j in line_4]
ys = [j[1] for j in line_4]
plt.plot(xs,ys,color='b',label='Ratio_4')
plt.scatter(xs,ys,color='b')

xs = [j[0]/100000 for j in line_6]
ys = [j[1] for j in line_6]
plt.plot(xs,ys,color='g',label='Ratio_6')
plt.scatter(xs,ys,color='g')

plt.legend()
plt.ticklabel_format(axis='x',scilimits=(0,10))
plt.xlabel("Number of Nodes ( x 10$^5$ )")
plt.ylabel("Time(s)")
#plt.savefig("plots/exp-3-scalability.png",dpi=150)
plt.show()