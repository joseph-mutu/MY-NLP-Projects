import vocabulary as v
import numpy as np

class SurnameVectorizer(object):
    """从数据集中实例化 vocabulary
    - 
    """
    def __init__(self, surname_vocab = None, nation_vocab = None):
        """
        surname_vocab: vocabulary 的实例化对象，针对 surname 建立
        nation_vocab: vocabulary 的实例化对象, 针对 nationality 建立
        """
        self.surname_vocab = surname_vocab
        self.nation_vocab = nation_vocab
    
    def vectorize(self,surname):
        """将 surname 向量化
        输入:
            surname(string): surname 的字符串
        输出:
            向量(list)
        """

        vocab = self.surname_vocab
        one_hot = np.zeros(vocab.__len__(), dtype=np.float32)

        for char in surname:
            one_hot[vocab.lookup_token(char)] = 1
        
        return one_hot
    
    def get_vector_size(self):
        return len(self.surname_vocab._token_to_idx)
    
    @classmethod
    def from_dataframe(cls,dataset):
        """从数据集中实例化 vectorizer
        输入:
            dataset(pd.DataFrame): 包含 surname 以及 nationality
        输出:
            vectorizer 的实例
        """
        surname_vocab = v.Vocabulary(add_unk = True, unk_token = '@')
        nation_vocab = v.Vocabulary(add_unk = False)

        for _, row in dataset.iterrows():
            for char in row.surname:
                surname_vocab.add_token(char)
            nation_vocab.add_token(row.nationality)
        
        return cls(surname_vocab,nation_vocab)
    
    def to_serializable(self):
        """将 vectorizer 序列化，返回一个字典
        """
        return {
            'surname_vocab': self.surname_vocab.to_serializable,
            'nation_vocab': self.nation_vocab.to_serializable
        }
    
    @classmethod
    def from_serializable(cls,contents):
        """从序列化的字典中生成 vectorizer 的实例
        输入:
            contents: 字典，to_serializable 的返回值
        输出:
            vectorizer 的实例
        """
        surname_vocab = v.Vocabulary.from_serializable(contents['surname_vocab'])
        nation_vocab = v.Vocabulary.from_serializable(contents['nation_vocab'])
        return cls(surname_vocab = surname_vocab, nation_vocab = nation_vocab)


    


    
