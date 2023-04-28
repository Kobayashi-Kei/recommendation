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
from recom import allPaperDataClass, PaperDataClass

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

for title, paper in allPaperData.paperDict.items():
    # ターゲット論文のみ抽出
    if paper['test'] != 1:
        continue

    randomIndexList = []

    for citeTitle in paper['cite']:
        count += 1
        
        # 引用関係有り
        simDict = {}
        for key in labelList:
            # ターゲット論文にも相手論文にもその観点に分類されたアブスト文があるなら
            if labeledAbstDict[title][key] and labeledAbstDict[citeTitle][key]:
                labelCount[key] += 1
                if method == 'tf-idf':
                    paperVector = vectorizer.transform(
                        [labeledAbstDict[paper["title"]][key]]).toarray().tolist()[0]
                    opponentPaperVector = vectorizer.transform(
                        [labeledAbstDict[citeTitle][key]]).toarray().tolist()[0]
                else:
                    paperVector = labeledAbstDict[title][key]
                    opponentPaperVector = labeledAbstDict[citeTitle][key]
                
                cosSim = cosine_similarity([paperVector], [opponentPaperVector])
                simDict[key] = cosSim[0][0]
                # euDis = distance.euclidean(paperVector, citePaperVector)
                # simDict[key] = euDis 
                
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
            citationEntire.append(1)
        else:
            citationEntire.append(0)
        
        

        maxKey = max(simDict, key=simDict.get)
        maxSimLabelCount[maxKey] += 1
        
        
        # 引用関係無し
        simDictNotCite = {}
        
        while True:
            randomIndex = int(randomNumberList[randomCount] * len(allPaperData.paperList))
            randomCount += 1
            if not randomIndex in randomIndexList and\
                not allPaperData.paperList[randomIndex]["title"] == title:
                    randomIndexList.append(randomIndex)
                    break
                
        for key in labelList:
            # ターゲット論文にも相手論文にもその観点に分類されたアブスト文があるなら
            targetLabel = labeledAbstDict[title][key]
            opponentLabel = labeledAbstDict[allPaperData.paperList[randomIndex]["title"]][key]
            if targetLabel and opponentLabel:
                labelCount[key] += 1
                if method == 'tf-idf':
                    paperVector = vectorizer.transform(
                        [targetLabel]).toarray().tolist()[0]
                    opponentPaperVector = vectorizer.transform(
                        [opponentLabel]).toarray().tolist()[0]
                else:
                    paperVector = targetLabel
                    opponentPaperVector = opponentLabel
                
                cosSim = cosine_similarity([paperVector], [opponentPaperVector])
                simDictNotCite[key] = cosSim[0][0]
                # euDis = distance.euclidean(paperVector, citePaperVector)
                # simDict[key] = euDis

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
            notCitationEntire.append(1)
        else:
            notCitationEntire.append(0)
        
        # TODO 
        # 1. 上のcosSimのしきい値をどうするか->手作業で0.739に決定
        # 2. プロットした点を見てそのアブストの文が見たい->やり方わからん
        
        
        # # ケース1
        # if 'title' in simDict \
        #     and 'bg' in simDict\
        #     and 'obj' in simDict:
        #     titleSimList.append(simDict['title'])
        #     bgSimList.append(simDict['bg'])
        #     objSimList.append(simDict['obj'])

        # # ケース2
        # if 'title' in simDict \
        #     and 'bg' in simDict\
        #     and 'method' in simDict:
        #     bgSimList.append(simDict['title'])
        #     objSimList.append(simDict['bg'])
        #     resSimList.append(simDict['method'])

        # # ケース3
        # if 'title' in simDict \
        #     and 'bg' in simDict\
        #     and 'result' in simDict:
        #     titleSimList.append(simDict['title'])
        #     bgSimList.append(simDict['bg'])
        #     resSimList.append(simDict['result'])

        # # ケース4
        # if 'bg' in simDict \
        #     and 'obj' in simDict\
        #     and 'method' in simDict:
        #     bgSimList.append(simDict['bg'])
        #     objSimList.append(simDict['obj'])
        #     methodSimList.append(simDict['method'])

        # # ケース5
        # if 'bg' in simDict \
        #     and 'obj' in simDict\
        #     and 'result' in simDict:
        #     bgSimList.append(simDict['bg'])
        #     objSimList.append(simDict['obj'])
        #     resSimList.append(simDict['result'])
        
        # ケース6
        if 'bg' in simDict \
            and 'method' in simDict\
            and 'res' in simDict:
            bgSimList['cite'].append(simDict['bg'])
            methodSimList['cite'].append(simDict['method'])
            resSimList['cite'].append(simDict['res'])
        
        # これ、引用関係かランダムかどちらかのみがプロットされることがあるじゃん、
        # それじゃターゲット論文をあわせられていないじゃん。
        # ラベルが無いことがあるから、ランダムに選ぶ方はラベルがあるものからランダムに選んだ方がいいのかな
        # ついでにexecを使って、ケースでコメントアウトせずにいい感じに3変数指定できるようにプログラムも書き換えたい
        # あとは全体でみた時のやつでプロットするマークを変える
        if 'bg' in simDictNotCite \
            and 'method' in simDictNotCite\
            and 'res' in simDictNotCite:
            bgSimList['notCite'].append(simDictNotCite['bg'])
            methodSimList['notCite'].append(simDictNotCite['method'])
            resSimList['notCite'].append(simDictNotCite['res'])

"""
集計結果を出力
"""
print("count", count)
print('labelCount:', labelCount)
print('maxCosSimLabelCount:', maxSimLabelCount)


"""
3次元で可視化
"""
# notebookなら
# %matplotlib notebook

# 描画エリアの作成
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# グラフタイトルを設定
# ax.set_title('',size=20)

# 軸ラベルのサイズと色を設定
# 散布図の作成

# ケース6
ax.set_xlabel("bg", size=15, color="black")
ax.set_ylabel("method", size=15, color="black")
ax.set_zlabel("res", size=15, color="black")
ax.scatter(bgSimList['cite'], methodSimList['cite'], resSimList['cite'], s=20, c="red")
ax.scatter(bgSimList['notCite'], methodSimList['notCite'], resSimList['notCite'], s=20, c="blue")

# 描画
plt.show()

fig.savefig("image/visualDist-3dim-" + method + ".png")
