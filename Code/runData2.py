import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel

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
#train classifier
ntopics    =  100
lda        = ldaTopicModel(n_topics = ntopics,alpha= (2./ntopics) * np.ones(ntopics),
                                        beta=0.01*np.ones(nWords))
#print X

fname = "ldaUCI_%dtopics.10Iter.pickle"%ntopics
############ TO SAVE, UNCOMMENT
lda.fit(X)
f = open(fname,"wb")
pickle.dump(lda,f)
f.close()


############ TO TEST, UNCOMMENT
f = open(fname,"rb")
lda = pickle.load(f)
f.close()
#
nwrd = 20
idx = np.argsort(-lda.wordsInTopic)
for i in range(ntopics):
    print "topic %d:"%i
    print wordlist[idx[i,:nwrd]],
##
#print lda.topicsInDoc
#
#
#
