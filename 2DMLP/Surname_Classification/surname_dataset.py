from torch.utils.data import Dataset, DataLoader
import pandas as pd
import vectorizer as vz
import json
import torch as t

class SurnameDataset(Dataset):
    def __init__(self,surname_df, vectorizer):
        """initialize
        
        Arguments:
            Dataset {the submodule of torch.nn} -- [description]
            surname_df {pd.DataFrame} -- [preprocessed dataframe]
            vectorizer {Vectorizer} -- [the instantiate of Vectorizer(self-defined class)]
        """
        self.surname_df = surname_df 
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[surname_df.split == "train"]
        self.train_size = len(self.train_df)

        self.test_df = self.surname_df[self.surname_df.split == "test"]
        self.test_size = len(self.test_df)

        self.val_df = self.surname_df[self.surname_df.split == "val"]
        self.val_size = len(self.val_df)

        self._lookup_dic = {
            "train":(self.train_df,self.train_size),
            "test":(self.test_df,self.test_size),
            "val":(self.val_df, self.val_size)
        }

        self.set_split("train")

        # set class weights due to the different number of data in each class

        #class_counts is the dictionary whose format is {"nationality_1": count, "nationality_2": count}
        class_counts = self.surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            """Self-defined sort key
            
            Arguments:
                item {[tuple]} -- [a tuple comes from class_counts.items()]
            
            Returns:
                [int] -- [corresponding nationality's index in the nation_vocab]
            """
            return self._vectorizer.nation_vocab.lookup_token(item[0])
        
        sorted_class_count = sorted(class_counts.items(), key = sort_key)
        frequencies = [count for _,count in sorted_class_count]
        self.class_weights = t.tensor([1.0 /frequency for frequency in frequencies],dtype = t.float32)

    def save_vectorizer(self,path):
        """save the vectorizer from a instantiation
        
        Arguments:
            path {[string]} -- [the file path to store the vectorizer]
        """
        with open(path,'w') as fp:
            # self._vectorizer is an instantiation of SurnameVectorizer class, to_serializable will return a dict
            json.dump(self._vectorizer.to_serializable(), fp)
    
    @classmethod
    def load_vectorizer(cls,path):
        """use json.load() to load an existing vectorizer
        
        Arguments:
            path {[string]} -- [file path to load the vectorizer]
        """
        with open(path,'r') as fp:
            return vz.SurnameVectorizer.from_serializable(json.load(fp))

    @classmethod
    def load_dataset_and_make_vectorizer(cls,file_path):
        """load the dataset and make the vectorizer by calling SurnameVectorizer.from_dataframe
        
        Arguments:
            file_path {[string]} -- [the file path of the dataset]
        
        Returns:
            [SurnameDataset] -- [return the instantiation of SurnameDataset]
        """
        surname_df = pd.read_csv(file_path)
        vectorizer = vz.SurnameVectorizer.from_dataframe(surname_df)
        return cls(surname_df,vectorizer)
    
    @classmethod 
    def load_dataset_and_load_vectorizer(cls,path_dataset,path_vectorizer): 
        surname_df = pd.read_csv(path_dataset)
        vectorizer = SurnameDataset.load_vectorizer(path_vectorizer)
        
        return cls(surname_df,vectorizer)

    def save_vectorizer(self,vectorizer_path):
        """Save the vectorizer from an instantiation of SurnameDataset class
        
        Arguments:
            vectorizer_path {[string]} -- [the path to store the vectorizer]
        """

        with open(vectorizer_path,'r') as fp:
            json.dump(self._vectorizer, fp)
    
    def get_vectorizer(self):
        """return the self._vectorizer
        """
        return self._vectorizer
    
    def set_split(self,split = 'train'):
        """Select the splits in the dataframe. It serves for the following __getitem__() and __len__()
        
        Arguments:
            split {[string]} -- [is 'train' by default,should be 'train','test','val]
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dic[self._target_split]
    
    def __getitem__(self,index):
        """the main entry point for Pytorch Dataset. Return the data and its label based on the index
        
        Arguments:
            index {[int]} -- [the index to the data]

        Returns:
            [dictionary] -- [the dictionary includes data and corresponding label]
        """

        row = self._target_df.iloc[index]

        data = self._vectorizer.vectorize(row.surname)

        label = self._vectorizer.nation_vocab._token_to_idx[row.nationality]

        return {
            'surname':data,
            'nation':label
        }

    def __len__(self):
        """returns the length of the target dataframe. Serve for Pytorch Dataset
        
        Returns:
            [int] -- [the length of the target dataframe]
        """
        return self._target_size
    
    def get_num_batches(self,batch_size):
        """get a number of batches
        
        Arguments:
            batch_size {int} -- [the number of data included in a batch]
        
        Returns:
            [int] -- [rounded number of batches]
        """
        return self._target_size // batch_size
        

def generate_batches(dataset,batch_size, shuffle = True,drop_last = True, device = 'cpu'):


    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)

    out_dict = {}

    for data_dict in dataloader:
        # the format of data_dict is {'surname':[[vector_1],[vector_2],[vector_3]],'nation':[label_1],[label_2],[label_3]}
        out_data_dict = {}
        for name, tensor in data_dict.items():
            # items will return a tuple that includes each key-value pair in the dictionary 
            out_data_dict[name] = tensor.to(device)
        yield out_data_dict
        
            
            

        
        
            




