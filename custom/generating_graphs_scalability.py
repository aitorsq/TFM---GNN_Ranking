from utils import *
random.seed(1)

'''

This code generates different ER graphs for scalability test
In order to keep the same data sctructure used we generate the centrality dictionary with 0 bet for all the nodes for avoiding computations since the bet value is not relevant for this experiment

'''


# Generating some Erdos-Renyi graphs for scalability tests

nodes = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
edges = [2,4,6]

for n in nodes:
    for e in edges:
        list_bet_data = list()
        p = e/n

        print(f"Generating ER graph: {n} nodes, {e} avg edges")
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(n,p = p,directed = True)
        if nx.number_of_isolates(g_nx)>0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = {j:0 for j in g_nkit.iterNodes()} # We set the betweenness data to 0 sinnce we don't need it for this experiment
        list_bet_data.append([g_nx,bet_dict])

        fname_bet = f"./graphs_scalability_test/ER_1_graph_{n}_nodes_{e}_edges.pickle"

        with open(fname_bet,"wb") as fopen:
            pickle.dump(list_bet_data,fopen)

print("Finished")