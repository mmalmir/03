import pprint
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ldaTopicModel import ldaTopicModel

#load classic data
classic    = sio.loadmat("../Data/Classic/classic400.mat")
wordlist   = classic["classicwordlist"]
wordFreq   = classic["classic400"]
trueLabels = classic["truelabels"][0]
wordFreq   = wordFreq.toarray()
lda        = ldaTopicModel(n_topics = 3)
X          = wordFreq
lda.fit(X)