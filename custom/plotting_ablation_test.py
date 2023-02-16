import matplotlib.pyplot as plt


xs = [1,2,3,4,5,6,7]
ys_1 = [0.9709, 0.9722,	0.9739,	0.9745,	0.9745,	0.9743,	0.9745]
e_1 =  [0.0016, 0.0016,	0.0019,	0.0021,	0.0023,	0.0023,	0.002]
ys_2 = [0.8792, 0.8916,	0.8973,	0.8993,	0.8995,	0.901	,0.9017]
e_2 =  [0.0375, 0.0293,	0.0238,	0.0231,	0.0208,	0.0203,	0.0191]
ys_3 = [0.8693, 0.884	,0.8917	,0.8939	,0.8953	,0.8975	,0.8989]
e_3 =  [0.035, 0.0231,	0.0168,	0.0147,	0.0141,	0.0129,	0.0117]
plt.errorbar(xs,ys_1,e_1)
plt.scatter(xs,ys_1,c='b')
plt.plot(xs,ys_1,c='b',label='Scale free')

plt.errorbar(xs,ys_2,e_2,c='brown')
plt.scatter(xs,ys_2,c='brown')
plt.plot(xs,ys_2,c='brown',label='Edor-Renyi')

plt.errorbar(xs,ys_3,e_3,c='g')
plt.scatter(xs,ys_3,c='g')
plt.plot(xs,ys_3,c='g',label='Gaussian Random Partition')

plt.ylabel("KT Score")
plt.xlabel("No. of layers")
plt.legend()
plt.ylim([0.3,1])
plt.show()