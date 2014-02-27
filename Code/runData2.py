import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel
from sklearn import manifold

#load uci data
data = sio.loadmat("../Data/dataset.mat")
data = data["dataset"]
#print data
ndocs = data[0,0]
nWords = data[1,0]
X = np.zeros([ndocs,nWords])
for i in range(3,data.shape[0]):
    doc,w,f = data[i,:]
    X[doc-1,w-1] = f

wordlist = []
f = open("../Data/vocab.kos.txt","rt")
for line in f:
    wordlist.append(line)
f.close()
wordlist = np.asarray(wordlist)

ntopics    = 10
nIter      = 1
alpha      = 0.1
beta       = .2
train      = True
lda        = ldaTopicModel(n_topics = ntopics,alpha= alpha * np.ones(ntopics),
                                        beta=beta*np.ones(nWords),
                                        nIter=nIter)

fname = "uci_alpha%0.3f_beta%0.3f_iter%d.pickle"%(alpha,beta,nIter)

if train:
    lda.fit(X)
    f = open(fname,"wb")
    pickle.dump(lda,f)
    f.close()


############ TO TEST, UNCOMMENT
f = open(fname,"rb")
lda = pickle.load(f)
f.close()

nwrd = 20
idx = np.argsort(-lda.wordsInTopic)
for i in range(ntopics):
    print "topic %d:"%i
    print wordlist[idx[0,:nwrd]],"\n"

print "MDS..."

theta = lda.topicsInDoc
theta = theta / np.tile(theta.sum(axis=1).reshape([-1,1]),[1,ntopics])


seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed)
pos = mds.fit(theta).embedding_

nmds = manifold.MDS(n_components=3, metric=False, max_iter=3000, eps=1e-12,
                    random_state=seed, n_jobs=1,n_init=1)
theta = nmds.fit_transform(theta, init=pos)


ax.scatter(theta[topic1Idx,0], theta[topic1Idx,1], theta[topic1Idx,2], c='r',label='True label=1')
ax.scatter(theta[topic2Idx,0], theta[topic2Idx,1], theta[topic2Idx,2], c='g',label='True label=2')
ax.scatter(theta[topic3Idx,0], theta[topic3Idx,1], theta[topic3Idx,2], c='b',label='True label=3')

handles, labels = ax.get_legend_handles_labels()
ax.legend()

#ax.legend([h1,h2,h3], ["True label=1","True label=2","True label=3"])


ax.set_xlabel("Topic 1")
ax.set_ylabel("Topic 2")
ax.set_zlabel("Topic 3")

#plt.show()


#calculate the percent correct