import json
from tools import arg_parse_from_commandline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from ranx import Qrels, Run, evaluate, compare
from tqdm import tqdm
import math
import numpy as np
import datetime
import json
import statistics
import collections
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from scipy.spatial import distance
import random


"""
12/20
    ターゲット論文を固定して、引用論文とのクラスごとの距離を可視化する。
"""


dt_now = datetime.datetime.now()
"""
Main
"""
"""
データセットのパスなどを代入
"""
args = arg_parse_from_commandline(['method'])
method = args.method

if method != 'tf-idf' and \
    method != 'bow' and \
    method != 'bm25' and \
    method != 'bert' and \
    method != 'scibert' and \
    method != 'specter' and \
    method != 'all':
        print("Methodの引数が間違っています")
        exit()


#dataPath = './data/axcell/small/axcellRecomData-old-includeLabeled.json'
dataPath = 'data/axcell/mini/axcellRecomData-old-includeLabeled.json'
#dataPath = 'data/axcell/axcellRecomData-includeLabeled.json'

#embPath = 'data/axcell/small/axcellRecomData-old-includeEmb-includeLabelEmb.json'
embPath = 'data/axcell/mini/axcellRecomData-old-includeEmb-includeLabelEmb.json'
#embPath = 'data/axcell/axcellRecomData-includeEmb-includeLabelEmb.json'

"""
データ
・論文データ全体 allPaperData
・テストデータ testPaperData

データ構造
paper {
    "id":1,
    "title": "On the Use of Context for Predicting Citation Worthiness of Sentences in Scholarly Articles",
    "abstract": "In this paper, we study the importance of context in predicting the citation worthiness of sentences in scholarly articles. We formulate this problem as a sequence labeling task solved using a hierarchical BiLSTM model. We contribute a new benchmark dataset containing over two million sentences and their corresponding labels. We preserve the sentence order in this dataset and perform document-level train/test splits, which importantly allows incorporating contextual information in the modeling process. We evaluate the proposed approach on three benchmark datasets. Our results quantify the benefits of using context and contextual embeddings for citation worthiness. Lastly, through error analysis, we provide insights into cases where context plays an essential role in predicting citation worthiness.",
    "cite":[2],
    "test": 1,
    "abstract_bert_embedding": [
        ...
        ]
}
"""

"""
データのロード・整形
"""
# データ辞書の用意
allPaperData = {
    'dataList':[],  # paperのリスト
    'abstractList':[], # paperから取り出したアブストラクトのリスト
                       # ※ これ以降のリストは全てdataListの順番を保持している
    'backgroundList':[], # アブストラクト分類結果のbackgroundの文章(or埋め込み)のリスト
    'objectiveList':[], # 同上
    'methodList':[], # 同上
    'resultList':[], # 同上
    'otherList':[], # 同上
    'titleList':[], # 同上
    'testDataIndex':[], # testデータならそのindexを格納
    'titleToIndex':{}
}
testPaperData = {
    'dataList':[], # 'test'==1 のpaperのリスト
    'allDataIndex':[], # alldataIndex[i]はdataList[i]のallPaperDataにおけるindexとなる
    'abstractList':[],
    'embeddingList':[],
} 

# データセットをロード
with open(dataPath, 'r') as f:
    allPaperData['dataList'] = json.load(f)

# アブストラクト毎のemmbeddingをロード
with open(embPath, 'r') as f:
    allEmbData = json.load(f)

# Vectorizerに合うようにアブストラクトのみをリストに抽出
for paper in allPaperData['dataList']:
    allPaperData['abstractList'].append(paper['abstract'])

# 予測結果の分析のため、タイトルをキーとして、indexをバリューとする辞書を生成
for i, paper in enumerate(allPaperData['dataList']):
    allPaperData['titleToIndex'][paper['title']] = i
    
labelList = ['title', 'backgroundText', 'objectiveText', 'methodText', 'resultText']
labelListKeyList = ['titleList', 'backgroundList', 'objectiveList', 'methodList', 'resultList']
# 分類されたアブストラクトごとにリストに抽出
if 'bert' in method or 'specter' in method:
    #labelList = ['title', 'backgroundText', 'objectiveText', 'methodText', 'resultText', 'otherText']
    #labelListKeyList = ['titleList', 'backgroundList', 'objectiveList', 'methodList', 'resultList', 'otherList']
    labelList = ['title', 'backgroundText', 'objectiveText', 'methodText', 'resultText']
    labelListKeyList = ['titleList', 'backgroundList', 'objectiveList', 'methodList', 'resultList']
    for label in labelList:
        listKey = label.replace('Text', 'List')
        if label == 'title': listKey = 'titleList'
        # BERT or SciBERT or SPECTER埋め込みをリストに抽出
        key = 'labeled_abstract_' + method + '_embedding'
        for paper in allEmbData:
            allPaperData[listKey].append(paper[key][label])
else:
    labelList = ['backgroundText', 'objectiveText', 'methodText', 'resultText', 'otherText']
    labelListKeyList = ['backgroundList', 'objectiveList', 'methodList', 'resultList', 'otherList']
    for label in labelList:
        listKey = label.replace('Text', 'List')
        # 各ラベルの文章をリストに抽出
        for paper in allPaperData['dataList']:
            allPaperData[listKey].append(paper['labeledAbstract'][label])

fiveDimList = []
dimTextList = []
count = 0

distDict = {
    'title':[],
    'background':[],
    'objective': [],
    'method': [],
    'result':[]
}

if method == 'tf-idf':
    vectorizer = TfidfVectorizer()
    vectorizer.fit(allPaperData['abstractList'])
    labelListKeyList.append('title')
    maxSimLabelCount = {label: 0 for label in labelListKeyList}
    labelCount = {label: 0 for label in labelListKeyList}
    labelListKeyList.pop(len(labelListKeyList)-1)
else:
    maxSimLabelCount = {label: 0 for label in labelListKeyList}
    labelCount = {label: 0 for label in labelListKeyList}
for i, paper in enumerate(allPaperData['dataList']):
    if paper['test'] != 1: continue
    
    simList = []
    count += 1
    if not len(paper['cite']) >= 2:
        continue
    
    for citeTitle in paper['cite']:
        index = allPaperData['titleToIndex'][citeTitle]
        citePaper = allPaperData['dataList'][index]
        
        simDict = {}
        for key in labelListKeyList:
            
            if allPaperData[key][i] and allPaperData[key][index]:
                labelCount[key] += 1
                if method == 'tf-idf':
                    paperVector = vectorizer.transform([allPaperData[key][i]]).toarray().tolist()[0]
                    citePaperVector = vectorizer.transform([allPaperData[key][index]]).toarray().tolist()[0]
                else:
                    paperVector = allPaperData[key][i]
                    citePaperVector = allPaperData[key][index]
    
                cosSim = cosine_similarity([paperVector], [citePaperVector])
                #cosSim = distance.euclidean(paperVector, citePaperVector)
                simDict[key] = cosSim[0][0]
                #print(simDict)
                #simDict[key] = cosSim
        # title
        if method == 'tf-idf':
            paperVector = vectorizer.transform([paper['title']]).toarray().tolist()[0]
            citePaperVector = vectorizer.transform([citeTitle]).toarray().tolist()[0]
            labelCount['title'] += 1
            cosSim = cosine_similarity([paperVector], [citePaperVector])
            #cosSim = distance.euclidean(paperVector, citePaperVector)
            simDict['title'] = cosSim[0][0]
            #simDict['title'] = cosSim
        
        #print(simDict)
        maxKey = max(simDict, key=simDict.get)
        maxSimLabelCount[maxKey] += 1
        
        # 5次元で格納
        tmpList = []
        #notExist = 0
        notExist = random.random() * 20
         
        if 'title' in simDict:
            tmpList.append(simDict['title'])
            distDict['title'].append(simDict['title'])
            #print(simDict)
        elif 'titleList' in simDict:
            tmpList.append(simDict['titleList'])
            distDict['title'].append(simDict['titleList'])
            #print(simDict)
        else:
            tmpList.append(notExist)
        if 'backgroundList' in simDict:
            tmpList.append(simDict['backgroundList'])
            distDict['background'].append(simDict['backgroundList'])
        else:
            tmpList.append(notExist)
        if 'objectiveList' in simDict:
            tmpList.append(simDict['objectiveList'])
            distDict['objective'].append(simDict['objectiveList'])
        else:
            tmpList.append(notExist)
        if 'methodList' in simDict:
            tmpList.append(simDict['methodList'])
            distDict['method'].append(simDict['methodList'])
        else:
            tmpList.append(notExist)
        if 'resultList' in simDict:
            tmpList.append(simDict['resultList'])
            distDict['result'].append(simDict['resultList'])
        else:
            tmpList.append(notExist)
        fiveDimList.append(tmpList)
        tmpStr = ""
        for item in tmpList:
            tmpStr += '{:.2f}'.format(item) + ","
        tmpStr = tmpStr[:-1]
        dimTextList.append(tmpStr)
        
        if 'title' in simDict:
            simDict['titleList'] = simDict['title']
            del simDict['title']
            
        simList.append(simDict)
        
    fig = plt.figure()
    plt.title(method)
    colorDict = {
        'titleList':'red',
        'backgroundList':'blue',
        'objectiveList': 'pink',
        'methodList': 'green',
        'resultList':'black'
    }
    for key in labelListKeyList:    
        axis_array = np.arange(0, len(simList), step=1)
        plt.xticks(axis_array)
        
        for j, citePaperDist in enumerate(simList):
            if key in citePaperDist:
                plt.scatter(j, citePaperDist[key],c=colorDict[key])
    fig.savefig("image/visualDist-lockPaper-" + str(i) + "-" +method + ".png")
        
print("count", count)
print('labelCount:', labelCount)
print('maxCosSimLabelCount:', maxSimLabelCount)

#print(distDict)