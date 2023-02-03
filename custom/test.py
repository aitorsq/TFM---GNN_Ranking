import matplotlib.pyplot as plt
import pickle
nodes = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
edges = [2,4,6]
with open(f"auxresults.pickle","rb") as fopen:
            results = pickle.load(fopen)

for e in edges:
    xs = []
    ys = []
    if e == 2:
        c = 'r'
        l = 'ratio 2'
    elif e == 4:
        c = 'g'
        l = 'ratio 4'
    else:
        c = 'b'
        l = 'ratio 6'
    for j in results:
        if j["e"] == e:
            xs.append(j["n"])
            ys.append(j["t"])
    plt.plot(xs,ys,color=c,label=l)
    plt.scatter(xs,ys,color=c)
plt.legend()
plt.show()