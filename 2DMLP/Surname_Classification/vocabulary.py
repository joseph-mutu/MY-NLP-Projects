"""
Vocabulary 的用处
    - 建立字典：将 token 映射为 index
    - 建立字典：根据 index 找到 token
    - 将未知 token 识别为 unk
    - 将新的 token 添加进字典 
""" 

class Vocabulary(object):
    def __init__(self,add_unk = True, unk_token = '<unk>', token_to_idx = None):
        """
             add_unk: 是否定义 unk 来识别未知的 token
             unk_token: 未知 token 的表示
             token_to_idx: 将 token 映射为 index 的字典
        """
        self.add_unk = add_unk
        self.unk_token = unk_token

        if token_to_idx is None:
            self._token_to_idx = {}

        self._idx_to_token = {idx:token for idx,token in self._token_to_idx.items()}

        self.unk_idx = -1
        if self.add_unk:
            self.unk_idx = self.add_token(self.unk_token)
    
    def to_serializable(self):
        """将当前的 token_to_idx,add_unk,unk_token 以字典的形式返回
        """
        return {
            'add_unk': self.add_unk,
            'unk_token': self.unk_token,
            'token_to_idx': self._token_to_idx
        }
        
    @classmethod 
    def from_serializable(cls,contents):
        """从已经 to_serializable 的 Vocabulary 中创建一个实例
        contents:
            为一个字典，其中包含
            - add_unk
            - unk_token
            - token_to_idx
        """
        return cls(**contents)

    def lookup_token(self,token):
        """查询 token 在字典中的 index
        token:
            待查询的token
        """
        if self.add_unk:
            index = self._token_to_idx.get(token,self.unk_idx)
        else:
            index = self._token_to_idx[token]
        return index
        
    def lookup_idx(self,index):
        """查询 index 代表的 token
        index:
            待查询的 index    
        """
        if index not in self._idx_to_token:
            raise KeyError("The index {} is not in the Vocabulary".format(index))
        else:
            token = self._idx_to_token[index]
        return token
    
    def add_token(self,token):
        """将 token 添加进字典当中，并返回相应 index
        操作：
            - 检测 token 是否已经存在，是则直接返回 index
            - 若不存在，则更新两个字典
        """
        try:
            index = self._token_to_idx[token]
        except KeyError:    
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self,token_list):
        """将一个 token list 进行添加
            token_list: token 的 list
        """
        self.add_token(token for token in token_list)
    
    def __len__(self):
        """ 返回字典的长度
        """
        return len(self._token_to_idx)
    
    def __str__(self):
        """返回一共存在多少个实例
        """
        print("Vocabulary(size{})".format(len(self)))
    

