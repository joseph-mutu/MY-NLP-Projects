import numpy as np
from annoy import AnnoyIndex

class PreTrainedEmbedding():
    def __init__(self,word_to_index,word_vectors):
        """
        
        Arguments:
            word_to_index {[Dictionary]} -- [a pre-defined dictionary mapping words to integers]
            word_vectors {[np.ndarray]} -- [word vectors loaded from Glove or word2Vec]
        """
        self.word_to_index = word_to_index
        self.index_to_word = {index:word for word, index in self.word_to_index.items()}
        self.word_vectors = word_vectors


        # The following is to build the annoy object in order to get the approximate nearest vectors
        # please refer to github spotify/annoy for details
        self.index = AnnoyIndex(len(word_vectors[0]),metric = 'euclidean')
        print("Building Index")
        for _, i in self.word_to_index.items():
            self.index.add_item(i,self.word_vectors[i])
        self.index.build(50)
        print("Building Finished")

    @classmethod
    def from_embedding_file(cls,path):
        """Main entry of PreTrainedEmbedding
        
        Arguments:
            path {[string]} -- [the file path of word_embedding file]
        
        Returns:
            [object] -- [an instantiation of PreTrained Class]
        """
        word_to_index = {}
        word_vectors = []

        with open(path,encoding = 'utf-8') as fp:
            for line in fp.readlines():
                line = line.split(" ")

                word = line[0]
                word_to_index[word] = len(word_to_index)
                
                vec = np.array([float(x) for x in line[1:]])

                word_vectors.append(vec)
        return cls(word_to_index = word_to_index, word_vectors = word_vectors)

    def get_embedding(self,word):
        """get the embedding vector given a word
        
        Arguments:
            index {[int]} -- [the index of the word]
        """
        if self.word_to_index and self.word_vectors:
            return self.word_vectors[self.word_to_index[word]]
        else:
            print("can not map {} to integers".format(word))
    
    def get_closest_to_vector(self,vector,n = 1):
        """return n closest vectors given one vector
        
        Arguments:
            vector {[np.ndarray]} -- [word vector]
        
        Keyword Arguments:
            n {int} -- [the number of words to return] (default: {1})
        
        Returns:
            [list] -- [words list which is not ordered by distances]
        """
        # please refer to annoy github page for details
        indices = self.index.get_nns_by_vector(vector,n)
        return [self.index_to_word[i] for i in indices]
    
    def compute_and_print_analogy(self,word1,word2,word3):

        word1_vec = self.get_embedding(word1)
        word2_vec = self.get_embedding(word2)
        word3_vec = self.get_embedding(word3)

        spatial_relation = word2_vec - word1_vec
        word4_vec = word3_vec + spatial_relation

        # to prevent the closest words to word4_vec are word1, word2 and word3
        closest_words = self.get_closest_to_vector(word4_vec,n=4)
        existing_words = [word1,word2,word3]
        closest_words = [word for word in closest_words if word not in existing_words]

        if not len(closest_words):
            print("Can not find the closest word")
        
        for word4 in closest_words:
            print("{}:{}:{}:{}".format(word1,word2,word3,word4))

file_path = "data/glove.6B.50d.txt"

pretrained = PreTrainedEmbedding.from_embedding_file(file_path)
pretrained.compute_and_print_analogy('man','king','woman')




