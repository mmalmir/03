import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel
from sklearn import manifold
from sklearn.decomposition import PCA

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
nIter      = 100
alpha      = 0.1
beta       = .2
train      = True
lda        = ldaTopicModel(n_topics = ntopics,alpha= alpha * np.ones(ntopics),
                                        beta=beta*np.ones(nWords),
                                        nIter=nIter)

fname = "uci_alpha%0.3f_beta%0.3f_iter%d.pickle"%(alpha,beta,nIter)

f = open(fname,"rb")
lda = pickle.load(f)
f.close()


if train:
    lda.fit(X)
    f = open(fname,"wb")
    pickle.dump(lda,f)
    f.close()

#
############# TO TEST, UNCOMMENT
f = open(fname,"rb")
lda = pickle.load(f)
f.close()

nwrd = 20
idx = np.argsort(-lda.wordsInTopic)
for i in range(ntopics):
    print "topic %d:"%i
    print wordlist[idx[i,:nwrd]],"\n"

print "visualizing..."
print lda.meanHarmonic
#theta = lda.topicsInDoc
#theta = theta / np.tile(theta.sum(axis=1).reshape([-1,1]),[1,ntopics])


#seed = np.random.RandomState(seed=3)
#mds = manifold.MDS(n_components=3, max_iter=30, eps=1e-9, random_state=seed)
#pos = mds.fit(theta).embedding_
#
#nmds = manifold.MDS(n_components=3, metric=False, max_iter=30, eps=1e-12,
#                    random_state=seed, n_jobs=1,n_init=1)
#theta = nmds.fit_transform(theta, init=pos)

#theta = PCA(n_components=3).fit_transform(theta)
#theta = theta / np.tile(theta.sum(axis=1).reshape([-1,1]),[1,3])
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(theta[:,0], theta[:,1], theta[:,2], c='r',label='True label=1')

#handles, labels = ax.get_legend_handles_labels()
#ax.legend()

#ax.legend([h1,h2,h3], ["True label=1","True label=2","True label=3"])


#ax.set_xlabel("Topic 1")
#ax.set_ylabel("Topic 2")
#ax.set_zlabel("Topic 3")
#
#plt.show()


#calculate the percent correct