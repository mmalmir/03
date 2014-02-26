import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel

#load uci data
data = sio.loadmat("../Data/dataset.mat")
data = data["dataset"]
ndocs = data[0,-1]
nWords = data[1,-1]
X = np.zeros([ndocs,nWords])
for i in range(3,data.shape[0]):
    doc,w,f = data[i,:]
    X[doc-1,w-1] = f

#train classifier
ntopics    =  10
lda        = ldaTopicModel(n_topics = 20,alpha= (2./ntopics) * np.ones(ntopics),
                                        beta=0.01*np.ones(nWords))
print X
lda.fit(X)

fname = "ldaUCI_20topics_alpha2k_beta.01.pickle"
f = open(fname,"wb")
pickle.dump(lda,f)
f.close()

#f = open("lda400.pickle","rb")
#lda = pickle.load(f)
#f.close()
#
#nwrd = 20
#idx = np.argsort(-lda.wordsInTopic)
#print "topic 1:",wordlist[idx[0,:nwrd]],"\n"
#print "topic 2:",wordlist[idx[1,:nwrd]],"\n"
#print "topic 3:",wordlist[idx[2,:nwrd]],"\n"
#
#print wordFreq
#print lda.topicsInDoc
#
#
#
