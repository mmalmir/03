import pprint
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


#load classic data
classic    = sio.loadmat("../Data/Classic/classic400.mat")
wordlist   = classic["classicwordlist"]
wordFreq   = classic["classic400"]
trueLabels = classic["truelabels"][0]
wordFreq   = wordFreq.toarray()
rows,cols  = np.where(wordFreq!=0)
vals       = wordFreq[rows,cols]
ndoc,nword = wordFreq.shape
words      = np.arange(nword)
count1,count2,count3 = [],[],[]
for doc,word,freq in zip(rows,cols,vals):
    if trueLabels[doc]==1:
        for i in range(freq):
            count1.append(word)
    elif trueLabels[doc]==2:
        for i in range(freq):
            count2.append(word)
    elif trueLabels[doc]==3:
        for i in range(freq):
            count3.append(word)
#extract x,y
#count1 = np.asarray(count1).reshape([1,-1])
#count2 = np.asarray(count2).reshape([1,-1])
#count3 = np.asarray(count3).reshape([1,-1])

bins = np.arange(0,nword,30)

plt.figure()
#x = np.concatenate([count1,count2,count3],axis=0)
#print x.shape
plt.hist([count1,count2,count3], bins, histtype='bar', stacked=True, fill=True)
#plt.hist(count2, bins, histtype='bar', stacked=True, fill=True)
#plt.hist(count3, bins, histtype='bar', stacked=True, fill=True)
plt.show()