import matplotlib.pyplot as plt



xs = [4,8,12,16,20]

# scale free
#title = 'Scale-free'
#ys_1 = [0.9706,	0.9731,	0.9736,	0.9733,	0.9734]
#e_1 =  [0.002,	0.0015,	0.0011,	0.0014,	0.0014]
#ys_4 = [0.9758,	0.9766,	0.9765,	0.9764,	0.9762]
#e_4 =  [0.0018,	0.002,	0.0023,	0.0023,	0.0022]
#ys_7 = [0.9759,	0.9768,	0.9765,	0.9762,	0.9762]
#e_7 =  [0.0023,	0.0023,	0.0022,	0.0023,	0.0024]


# erdos renyi
title = 'Erdos-Renyi'
ys_1 = [0.8577, 0.8724, 0.8786, 0.8812, 0.8798]
e_1 =  [0.0595, 0.0437, 0.0387, 0.0417, 0.0401]
ys_4 = [0.869, 0.8739, 0.8787, 0.8635, 0.8449]
e_4 =  [0.0553, 0.0368, 0.051, 0.0751, 0.0714]
ys_7 = [0.8815, 0.8593, 0.8706, 0.8596, 0.8669]
e_7 =  [0.043, 0.0512, 0.0652, 0.0995, 0.0741]


# GRP
#title = 'Gaussian Random Partition'
#ys_1 = [0.6548	,0.7765	,0.8231	,0.775	,0.7485]
#e_1 =  [0.3383	,0.1815	,0.109	,0.1933	,0.227]
#ys_4 = [0.7252	,0.7747	,0.8346	,0.8196	,0.8419]
#e_4 =  [0.2601	,0.1777	,0.098	,0.116	,0.0911]
#ys_7 = [0.8169	,0.7813	,0.7612	,0.7388	,0.7603]
#e_7 =  [0.1013	,0.1781	,0.21	,0.2542	,0.2095]

plt.errorbar(xs[:],ys_1[:],e_1[:])
plt.scatter(xs[:],ys_1[:],c='b')
plt.plot(xs[:],ys_1[:],c='b',label='1-layered')

plt.errorbar(xs[:],ys_4[:],e_4[:],c='g')
plt.scatter(xs[:],ys_4[:],c='g')
plt.plot(xs[:],ys_4[:],c='g',label='4-layered')

plt.errorbar(xs[:],ys_7[:],e_7[:],c='darkred')
plt.scatter(xs[:],ys_7[:],c='darkred')
plt.plot(xs[:],ys_7[:],c='darkred',label='7-layered')

plt.ylabel("KT Score")
plt.xlabel("Embedding dimensions")
plt.xticks(xs,xs)
plt.legend(loc=4)
plt.title(title)
plt.ylim([0.7,1])
plt.show()