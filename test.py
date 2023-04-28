import os
from tqdm import tqdm
import time
import re
import numpy as np
import traceback

"""
print(os.path.basename(__file__))
print(type(os.path.basename(__file__)))



for i in tqdm(range(2)):
    time.sleep(1)

text = "  Attention  is all you   need  ".strip().lower()
print(re.sub('\s{2,}', ' ', text) == 'attention is all you need')


labelList = ['a', 'b', 'c']

dic = {v : [] for v in labelList}
print(dic)


matrix = [[1,2],[3,4],[5,6]]

print([[0]*len(matrix[0])]*len(matrix))
"""
"""
print([[]]*10)

from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np 

a = [[2, 15, 3, 25], False] 
b = [[5, 25, 8, 28], [2, 15, 3, 25]]

array_1 = np.array(a) 
array_2 = np.array(b) 
cos_sim = cosine_similarity(array_1 , array_2) 
print(cos_sim) 
# 結果: 
# [[0.9753719]] 
"""
"""
mergeSimMatrix = [[0]*4]*10
print(mergeSimMatrix)
"""
"""
import datetime

dt_now = datetime.datetime.now()

print(dt_now)
# 2019-02-04 21:04:15.412854

print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))
# 2019年02月04日 21:04:15

print(dt_now.strftime('%Y%m%d%H%M%S'))
# 20221110195414
"""
"""
print(__file__.split('/')[-1])
"""
"""
a = np.array([1,2,3])
print(a*3)
"""

# try: 
#     str = 1 + "sss"
# except:
#     # errorStr = str(e)
#     trace = traceback.format_exc()
#     # print(type(errorStr),  e)
#     print(type(trace), trace)
    
# class A:
#     def __init__(self):
#         self.poko = "a"

# a = A()

# print(a.poko)
# print(a['poko'])

import random
import json

# randomKoteiList = []
# for i in range(10000):
#     randomKoteiList.append(random.random())

# with open('dataserver/randomNumberList.json', 'w')as f:
#     json.dump(randomKoteiList, f)

# tmp = np.array([[10,20,5], [1,2,3]])

# print(1/tmp)

# import time

# while True:
#     print(1)
#     time.sleep(2)

# a = [[1,0.85752068,0.75053292,0.85904449,0.79128929,0.84352999
# ,0.81263551,0.85624601,0.76320389,0.69134248,0.61885847,0.69871558
# ,0.71955423,0.72843263,0.73880132,0.72640684,0.79222901,0.82164307
# ,0.8119475, 0.85082246]]

# print(np.sort(a))
# print(np.argsort(a))


from sklearn import preprocessing

a = [np.nan, 1,2,3]
print(preprocessing.scale(a))