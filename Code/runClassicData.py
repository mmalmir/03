import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel
from mpl_toolkits.mplot3d import Axes3D


#load classic data
classic    = sio.loadmat("../Data/Classic/classic400.mat")
wordlist   = classic["classicwordlist"]
wordFreq   = classic["classic400"]
trueLabels = classic["truelabels"][0]
nWords     = 6205
wordFreq   = wordFreq.toarray()
lda        = ldaTopicModel(n_topics = 3,alpha= (0.1) * np.ones(3),
                                        beta=2.*np.ones(nWords))
X          = wordFreq

fname = "lda400_alpha.1_beta2.pickle"

lda.fit(X)

f = open(fname,"wb")
pickle.dump(lda,f)
f.close()

f = open(fname,"rb")
lda = pickle.load(f)
f.close()

nwrd = 20
idx = np.argsort(-lda.wordsInTopic)
print "topic 1:",wordlist[idx[0,:nwrd]],"\n"
print "topic 2:",wordlist[idx[1,:nwrd]],"\n"
print "topic 3:",wordlist[idx[2,:nwrd]],"\n"

#print wordFreq
theta = lda.topicsInDoc
theta = theta / np.tile(theta.sum(axis=1).reshape([-1,1]),[1,3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#n = 100
#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zl, zh)
print trueLabels
topic1Idx = np.where(trueLabels==1)
topic2Idx = np.where(trueLabels==2)
topic3Idx = np.where(trueLabels==3)

h1 = ax.scatter(theta[topic1Idx,0], theta[topic1Idx,1], theta[topic1Idx,2], c='r',label="True label=1")
h2 = ax.scatter(theta[topic2Idx,0], theta[topic2Idx,1], theta[topic2Idx,2], c='g',label="True label=2")
h3 = ax.scatter(theta[topic3Idx,0], theta[topic3Idx,1], theta[topic3Idx,2], c='b',label="True label=3")

#handles, labels = ax.get_legend_handles_labels()
#ax.legend([h1,h2,h3], ["True label=1","True label=2","True label=3"])


ax.set_xlabel("Topic 1")
ax.set_ylabel("Topic 2")
ax.set_zlabel("Topic 3")

plt.show()








