import re
import pandas as pd
from argparse import Namespace
from collections import defaultdict
import numpy as np

args = Namespace(
    seed = 2,
    train_proportion = 0.7,
    val_proportion = 0.15,
    test_proportion = 0.15,

    # the path of data
    raw_dataset = "data/surnames.csv",
    output_csv = "data/surnames_with_splits"
)

row_data = pd.read_csv(args.raw_dataset,header = 0)

# Split the dataset
by_nationality = defaultdict(list)

# 根据数据的 nationality 构建一个 list 列表
for _,row in row_data.iterrows():
    """
    下式形式为:
        data_dict = {"English":[{'surname':woordford,'nation':English},{'surname':cote,'nationality':French}..]}
        一个数据点为一个列表，具有相同 nationality 的数据被添加进同一个列表
    Note:
        iterrows 时返回的是 Series
    """

    by_nationality[row.nationality].append(row.to_dict())
    
"""
# Shuffle the data and assign the data with new attributes
# train, test and validation
"""

final_list = []

for _,item in sorted(by_nationality.items()):
    """
        _ 表示 nationality,也就是字典的键
        item 表示字典的值，也就是所有为该 nationality 的数据, 类型为 列表
        items() 返回一个元组
    """
    np.random.shuffle(item) # numpy 的 shuffle 对列表进行打乱操作   

    num_data = len(item)

    n_train = int(num_data * args.train_proportion)
    n_test = int(num_data * args.test_proportion)
    
    # 为每一个数据赋予一个新的属性 split 代表数据分类为 train, test 还是 validation
    for data in item[0:n_train]:
        data['split'] = 'train'
    for data in item[n_train+1:n_train+1 + n_test]:
        data['split'] = 'test'
    for data in item[n_train+1 + n_test:]:
        data['split'] = 'val'
    
    final_list.extend(item)

final_data = pd.DataFrame(final_list)
print(final_data.split.value_counts())
final_data.to_csv(args.output_csv)





