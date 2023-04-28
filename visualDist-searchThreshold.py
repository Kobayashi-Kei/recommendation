from mpl_toolkits.mplot3d import Axes3D
import json
from tools import arg_parse_from_commandline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from scipy.spatial import distance
import math
import random
from recom import allPaperDataClass

dt_now = datetime.datetime.now()

"""
手法
"""
args = arg_parse_from_commandline(['method'])
method = args.method

if method != 'tf-idf' and \
        method != 'bow' and \
        method != 'Bert' and \
        method != 'SciBert' and \
        method != 'Specter':
    print("Methodの引数が間違っています")
    exit()

"""
データ構造の定義
"""
allPaperData = allPaperDataClass()
testPaperData = PaperDataClass()

"""
データファイルの読み込み
"""
size = "medium"
# size = "small"

path = "dataserver/axcell/" + size + "/paperDict.json"
with open(path, 'r') as f:
    allPaperData.paperDict = json.load(f)

if method == 'tf-idf' or method == 'bow':
    path = "dataserver/axcell/" + size + "/labeledAbst.json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)
    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title
else:
    path = "dataserver/axcell/" + size + "/embLabel/labeledAbst" + method + ".json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)

    path = "dataserver/axcell/medium/embedding/titleAbst" + method + ".json"
    with open(path, 'r') as f:
        abstEmb = json.load(f)

"""
データの整形
"""
labelList = ['title', 'bg', 'obj', 'method', 'res']

for title, paper in allPaperData.paperDict.items():
    # Vectorizerに合うようにアブストラクトのみをリストに抽出
    allPaperData.abstList.append(paper["abstract"])

    # 分類されたアブストラクトごとにリストに抽出
    labelAbst = labeledAbstDict[paper["title"]]
    for label in labelList:
        allPaperData.labelList[label].append(labelAbst[label])

# 辞書をリストに変換
allPaperData.paperList = list(allPaperData.paperDict.values())

count = 0

# 採用する3観点分のリストを用意
titleSimList = {'cite': [],'notCite': []}
bgSimList = {'cite': [],'notCite': []}
objSimList = {'cite': [],'notCite': []}
methodSimList = {'cite': [],'notCite': []}
resSimList = {'cite': [],'notCite': []}

maxSimLabelCount = {label: 0 for label in labelList}
labelCount = {label: 0 for label in labelList}
if method == 'tf-idf':
    vectorizer = TfidfVectorizer()
    # TODO: tf-idfの語彙ってどう扱えばいいんだろうか
    vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])
    #vectorizer.fit(allPaperData.abstList)

# 乱数用
randomCount = 0
with open("dataserver/randomNumberList.json", 'r') as f:
    randomNumberList = json.load(f)
    
# アブスト全体で見た時に正解か不正解かを格納する
citationEntire = []
notCitationEntire = []
threshold = 0.739
correct = 0
error = 0

for title, paper in allPaperData.paperDict.items():
    # ターゲット論文のみ抽出
    if paper['test'] != 1:
        continue

    randomIndexList = []

    for citeTitle in paper['cite']:
        count += 1
        
        # アブスト全体で見た時に正解かどうか
        if method == 'tf-idf':
            paperVector = vectorizer.transform(
                [paper["abstract"]]).toarray().tolist()[0]
            opponentPaperVector = vectorizer.transform(
                [allPaperData.paperDict[citeTitle]["abstract"]]).toarray().tolist()[0]
        else:
            paperVector = abstEmb[title]
            opponentPaperVector = abstEmb[citeTitle]
        cosSim = cosine_similarity([paperVector], [opponentPaperVector])
        if cosSim >= threshold:
            correct += 1
        else:
            error += 1
        
        # 引用関係無し
        simDictNotCite = {}
        
        while True:
            randomIndex = int(randomNumberList[randomCount] * len(allPaperData.paperList))
            randomCount += 1
            if not randomIndex in randomIndexList and\
                not allPaperData.paperList[randomIndex]["title"] == title:
                    randomIndexList.append(randomIndex)
                    break

        # アブスト全体で見た時に正解かどうか
        if method == 'tf-idf':
            paperVector = vectorizer.transform(
                [paper["abstract"]]).toarray().tolist()[0]
            opponentPaperVector = vectorizer.transform(
                [allPaperData.paperDict[allPaperData.paperList[randomIndex]["title"]]["abstract"]]).toarray().tolist()[0]
        else:
            paperVector = abstEmb[title]
            opponentPaperVector = abstEmb[allPaperData.paperList[randomIndex]["title"]]
        cosSim = cosine_similarity([paperVector], [opponentPaperVector])
        if cosSim < threshold:
            correct += 1
        else:
            error += 1
    
print(correct)
print(error)