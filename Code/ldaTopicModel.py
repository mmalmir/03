import numpy as np

class ldaTopicModel:
    def __init__(self,**kwrds):
        self.nTopics = 3
        self.beta    = []
        self.alpha   = []
        self.set_params(**kwrds)

    def fit_transform(self,X,y=None,**kwrds):
        self.fit(X,y,**kwrds)
        return self.transform(X)

    def fit(self,X,y=None,**kwrds):
        self.set_params(**kwrds)
        self.trainCorpus(X)

    def set_params(self,**kwrds):
        for k in kwrds:
            if k=="n_topics":
                self.nTopics = kwrds[k]
            elif k=="beta":
                self.beta  = kwrds[k]
            elif k=="alpha":
                self.alpha = kwrds[k]



    def get_params(self,deep=False):
        return dict()
    
    def transform(self,X):
        return None

    #training code
    def trainCorpus(self,X):
        docs,words     = np.where(X!=0)
        freq           = X[docs,words]
        #maxfreq              = np.max(freq) #dimention?
        ndocs,dicSize  = X.shape
        topicForWord   = np.zeros(freq.sum()) #[t] topic
        #topicForword = np.zeros([ndocs,dicSize,maxfreq)
        #create count arrays
        wordsInTopic        = np.zeros([self.nTopics,dicSize]) # [k,t] number of words t and topic k

        #wordsInTopic       = np.zeros([self.nTopics,dicSize])
        topicsInDoc         = np.zeros([ndocs,self.nTopics])   # [m,k] number of topic k in document m
        totalWordsInTopic   = np.zeros(self.nTopics)           # [k] number of words with topic k
        totalTopicsInDoc    = np.zeros(ndocs)                  # [m] number of topics in document m
        cnt = 0
        #intialize counts
        unif = [1./self.nTopics]*self.nTopics
        print unif
        for d,w,f in zip(docs,words,freq):# document d with word w, with freq f
            for ii in range(f):
                z_m_n = np.random.multinomial(1,unif , size=1)[0]
                z_m_n = np.where(z_m_n==1)[0]
#                print z_m_n
                #z[d][w][ii]=np.where(np.random.mutinomial((1, [1/3.]*3, size=1) != 0)[1][0]
                wordsInTopic[z_m_n,w]    += 1
               #wordsInTopic[z[d][w][ii],w]] +=1
#                totalWordsInTopic[z_m_n] += 1
               #totalWordsInTopic[z[d][w][ii]] += 1
                topicForWord[cnt]        = z_m_n
               #topicForWord[d][w][ii] = z[d][w][ii]
                topicsInDoc[d,z_m_n]     += 1 #topic for document d
               #topicsInDoc[d,z[d][w][ii]]     += 1 #topic for document d
#                totalTopicsInDoc[d]     += 1
               #totalTopicsInDoc[d]     += 1
                cnt += 1
        converged = False
        print topicsInDoc
        for iiii in range(100):
            print iiii
        #do one round of gibbs sampling
        #update counts
            cnt = 0
            for d,w,f in zip(docs,words,freq):# document d with word w, with freq f
                for ii in range(f):
                    topicsInDoc[d,topicForWord[cnt]]     -= 1
                   #topicsInDoc[d,topicForWord[d][w][ii]]     -= 1
                    wordsInTopic[topicForWord[cnt],w]    -= 1
                   #wordsInTopic[topicForWord[d][w][ii],w]    -= 1
#                    totalTopicsInDoc[d]                  -= 1
                   #totalTopicsInDoc[d]                  -= 1
#                    totalWordsInTopic[topicForWord[cnt]] -= 1
                   #totalWordsInTopic[topicForWord[d][w][ii]] -= 1
                    prob = np.zeros(self.nTopics)
                    for k in range(self.nTopics):
                        prob[k] =  (wordsInTopic[k,w]+self.beta[w]) / np.sum(wordsInTopic[k,:]+self.beta)
                        prob[k] *= (topicsInDoc[d,k]+self.alpha[k]) / np.sum(topicsInDoc[d,:]+self.alpha)
                        #prob[k]=(wordsInTopic[topicForWord[d][w][ii],w]+self.beta[w])*(topicsInDoc[d,z[d][w][ii]]+self.alpha[d])
#                        prob[k] /= (totalTopicsInDoc[d]+self.alpha)*(totalWordsInTopic[k])
                    #prob[k]/= (topicsInDoc[d,z[d][w][ii]]+np.sum(self.alpha))*(totalWordsInTopic[topicForWord[d][w][ii]]+np.sum(self.beta))
                    prob = prob / prob.sum()
#                    print prob
#                    print prob
                    z_m_n = np.random.multinomial(1,prob, size=1)[0]
                    z_m_n = np.where(z_m_n==1)[0]
                    wordsInTopic[z_m_n,w]    += 1 #
                    #                totalWordsInTopic[z_m_n] += 1
                    # totalWordsInTopic[z_m_n] +=1
                    topicForWord[cnt]        = z_m_n
                    #topicForWord[d][w][ii] = z_m_n
                    topicsInDoc[d,z_m_n]     += 1 #topic for document d
                    #topicsInDoc[d,z_m_n]     += 1 #topic for document d
                    #                totalTopicsInDoc[d]     += 1
                    # totalTopicsInDoc[d] +=1
                    cnt += 1
            #check for convergence
            print topicsInDoc
#        print wordsInTopic
        self.topicsInDoc  = topicsInDoc
        self.wordsInTopic = wordsInTopic


