import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 10})


xs = range(1,8)

# scale free
#title = 'Scale-free'
ys_1 = [0.9734,	0.9752,	0.976,	0.9762,	0.9763,	0.9765,	0.9762]
e_1 =  [0.0014,	0.0017,	0.002,	0.0022,	0.0026,	0.0024,	0.0024]

# erdos renyi
#title = 'Erdos-Renyi'
#
#ys_1 = []
#e_1 =  []
ys_4 = [0.8798,	0.8841,	0.8862,	0.8449,	0.8524,	0.8509,	0.8669]
e_4 =  [0.0401,	0.0358,	0.0364,	0.0714,	0.096,	0.1018,	0.0741]
#ys_7 = [
#e_7 =  []


# GRP
#title = 'Gaussian Random Partition'
#
#ys_1 = []
#e_1 =  []
#ys_4 = []
#e_4 =  []
ys_7 = [0.7485,	0.7913,	0.8573,	0.8419,	0.8349,	0.8278,	0.7603]
e_7 =  [0.227,	0.1625, 0.069,	0.0911,	0.1071,	0.1099,	0.2095]

plt.errorbar(xs[:],ys_1[:],e_1[:])
plt.scatter(xs[:],ys_1[:],c='b')
plt.plot(xs[:],ys_1[:],c='b',label='Scalefree')

plt.errorbar(xs[:],ys_4[:],e_4[:],c='darkred')
plt.scatter(xs[:],ys_4[:],c='darkred')
plt.plot(xs[:],ys_4[:],c='darkred',label='Erdos-Renyi')

plt.errorbar(xs[:],ys_7[:],e_7[:],c='g')
plt.scatter(xs[:],ys_7[:],c='g')
plt.plot(xs[:],ys_7[:],c='g',label='Gaussian Random Partition')

plt.ylabel("KT Score")
plt.xlabel("No. of layers")
plt.xticks(xs,xs)
plt.legend()
#plt.title(title)
plt.ylim([0.6,1])
plt.show()