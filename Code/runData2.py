import pprint
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel

#load uci data
data = sio.loadmat("../Data/dataset.mat")
data = data["dataset"]
print data
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
ntopics    =  3
lda        = ldaTopicModel(n_topics = ntopics,alpha= (2./ntopics) * np.ones(ntopics),
                                        beta=0.01*np.ones(nWords))
#print X
lda.fit(X)

############ TO SAVE, UNCOMMENT

fname = "ldaUCI_3topics.10Iter.pickle"
f = open(fname,"wb")
pickle.dump(lda,f)
f.close()


############ TO TEST, UNCOMMENT
#f = open("ldaUCI_20topics_alpha2k_beta.01.pickle","rb")
#lda = pickle.load(f)
#f.close()
##
#nwrd = 20
#idx = np.argsort(-lda.wordsInTopic)
#print "topic 1:",wordlist[idx[0,:nwrd]],"\n"
#print "topic 2:",wordlist[idx[1,:nwrd]],"\n"
#print "topic 3:",wordlist[idx[2,:nwrd]],"\n"
#print "topic 4:",wordlist[idx[3,:nwrd]],"\n"
#print "topic 5:",wordlist[idx[4,:nwrd]],"\n"
#print "topic 6:",wordlist[idx[5,:nwrd]],"\n"
#print "topic 7:",wordlist[idx[6,:nwrd]],"\n"
#print "topic 8:",wordlist[idx[7,:nwrd]],"\n"
#print "topic 9:",wordlist[idx[8,:nwrd]],"\n"
#print "topic 10:",wordlist[idx[9,:nwrd]],"\n"
##
#print lda.topicsInDoc
#
#
#
