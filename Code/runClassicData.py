import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel

#load classic data
classic    = sio.loadmat("../Data/Classic/classic400.mat")
wordlist   = classic["classicwordlist"]
wordFreq   = classic["classic400"]
trueLabels = classic["truelabels"][0]
nWords     = 6205
wordFreq   = wordFreq.toarray()
lda        = ldaTopicModel(n_topics = 3,alpha= (2./3.) * np.ones(3),
                                        beta=0.01*np.ones(nWords))
X          = wordFreq

lda.fit(X)

fname = "lda400_alpha2k_beta.01.pickle"
f = open(fname,"wb")
pickle.dump(lda,f)
f.close()

#f = open(fname,"rb")
#lda = pickle.load(f)
#f.close()

nwrd = 20
idx = np.argsort(-lda.wordsInTopic)
print "topic 1:",wordlist[idx[0,:nwrd]],"\n"
print "topic 2:",wordlist[idx[1,:nwrd]],"\n"
print "topic 3:",wordlist[idx[2,:nwrd]],"\n"

print wordFreq
print lda.topicsInDoc



