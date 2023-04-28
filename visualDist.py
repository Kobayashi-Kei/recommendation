import json
from tools import arg_parse_from_commandline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
from scipy.spatial import distance
import random
import statistics
from recom import allPaperDataClass


"""
ターゲット論文と被引用論文との距離（類似度）をターゲット論文数分計算し、ヒストグラムで可視化する
"""
def main():
    dt_now = datetime.datetime.now()


    """
    手法
    """
    args = arg_parse_from_commandline(['method', 'simMethod'])
    method = args.method
    simMethod = args.simMethod

    if method != 'tf-idf' and \
            method != 'bow' and \
            method != 'Bert' and \
            method != 'SciBert' and \
            method != 'Specter':
        print("Methodの引数が間違っています")
        exit()

    if simMethod == "euclid":
        outputDir = "image/labelDist/euclid/"
    elif simMethod == "cos":
        outputDir = "image/labelDist/cos/"
    else:
        print("第2引数は euclid or cos としてください。")
        exit()
        

    """
    データ構造の定義
    """
    allPaperData = allPaperDataClass()

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

    """
    類似度測定
    """
    count = 0

    distListDict = {
        'title':[],
        'bg':[],
        'obj': [],
        'method': [],
        'res':[]
    }
    maxSimLabelCount = {label: 0 for label in labelList}
    labelCount = {label: 0 for label in labelList}

    notCiteDistListDict = {
        'title':[],
        'bg':[],
        'obj': [],
        'method': [],
        'res':[]
    }
    notCiteMaxSimLabelCount = {label: 0 for label in labelList}
    notCiteLabelCount = {label: 0 for label in labelList}



    # TF-IDFの計算のためのデータのfit
    if method == 'tf-idf':
        vectorizer = TfidfVectorizer()
        # TODO: tf-idfの語彙ってどう扱えばいいんだろうか
        # vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])
        vectorizer.fit(allPaperData.abstList)
        
    # 乱数用
    randomCount = 0
    with open("dataserver/randomNumberList.json", 'r') as f:
        randomNumberList = json.load(f)
        
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
                    
                    if simMethod == "cos":
                        sim = cosine_similarity([paperVector], [opponentPaperVector])[0][0]
                    elif simMethod == "euclid":
                        sim = distance.euclidean(paperVector, opponentPaperVector)
                    distListDict[key].append(sim)
                    simDict[key] = sim
                    
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
                    notCiteLabelCount[key] += 1
                    if method == 'tf-idf':
                        paperVector = vectorizer.transform(
                            [targetLabel]).toarray().tolist()[0]
                        opponentPaperVector = vectorizer.transform(
                            [opponentLabel]).toarray().tolist()[0]
                    else:
                        paperVector = targetLabel
                        opponentPaperVector = opponentLabel
                    
                    if simMethod == "cos":
                        sim = cosine_similarity([paperVector], [opponentPaperVector])[0][0]
                    elif simMethod == "euclid":
                        sim = distance.euclidean(paperVector, opponentPaperVector)
                    notCiteDistListDict[key].append(sim)
                    simDictNotCite[key] = sim
                                    
            maxKey = max(simDictNotCite, key=simDictNotCite.get)
            notCiteMaxSimLabelCount[maxKey] += 1

                    
                    
    print("count", count)
    print("--citaion--")
    print('labelCount:', labelCount)
    print('maxCosSimLabelCount:', maxSimLabelCount)
    print("--random--")
    print('notCiteLabelCount:', notCiteLabelCount)
    print('notCiteMaxCosSimLabelCount:', notCiteMaxSimLabelCount)
    print("==================")
    """
    # 
    # 1次元ごとに可視化
    #
    """
    import numpy as np

    for key in distListDict:
        
        fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # plot.titleでは、グラフのタイトルを指定しています。
        plt.title(method + "-" + key)
        
        print('-----', key, '-----')
        print("--random--")
        notciteData = np.array(notCiteDistListDict[key])
        print('median', '{:.2f}'.format(statistics.median(notciteData)))
        print('mean','{:.2f}'.format(statistics.mean(notciteData))) 
        # ax.hist(notciteData, bins=10, histtype='barstacked', ec='red')
        plt.hist(notciteData, bins=10, histtype='barstacked', alpha = 0.5, label='random')

        print("--citation--")
        citationData = np.array(distListDict[key])
        print('median', '{:.2f}'.format(statistics.median(citationData)))
        print('mean','{:.2f}'.format(statistics.mean(citationData))) 
        # ax.hist(citationData, bins=10, histtype='barstacked', ec='black')
        plt.hist(citationData, bins=10, histtype='barstacked', alpha = 0.5, label='citation')
        
        plt.legend(loc='upper left')
        # fig.savefig("image/visualDist-cosSim-" +method + "-" + key + ".png")
        fig.savefig(outputDir + method + "-" +key + ".png")

if __name__ == "__main__":
    main()