import numpy as np
import lda
# import lda.datasets



def load_node():
	data = []
	for line in file("node_doc.txt"):
		data.append(np.array([int(i) for i in line.replace("\n", "").split()]))
	return data


if __name__ == "__main__":
    X = load_node()
    # vocab = lda.datasets.load_reuters_vocab()
    # titles = lda.datasets.load_reuters_titles()
    model = lda.LDA(n_topics=39, n_iter=2, random_state=1)
    model.fit(X)
    model.topic_word_.tofile("topic_word.txt")
    # print model.nzw_
    
    
    '''
    topic_word = model.topic_word_  # model.components_ also works
    #print topic_word
    
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
    	# print topic_dist
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    '''