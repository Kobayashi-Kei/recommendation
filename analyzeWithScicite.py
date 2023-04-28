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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from recom import allPaperDataClass

"""
引用意図データセットSciCiteを用いた分析を行う
1. SciCiteにおけるターゲット論文と被引用論文との観点毎の類似度を算出する
2. それらの類似度が最大となる観点と、データにつけられた引用意図とが一致するかを検証する
3. 結果をAccuracyと混同行列で出力する
"""
def main():
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

    allPaperData = allPaperDataClass()

    """
    データファイルの読み込み
    """
    size = "test"
    
    path = "dataserver/scicite/" + size + "/main.json"
    with open(path, 'r') as f:
        mainData = json.load(f)

    path = "dataserver/scicite/" + size + "/paperDict.json"
    with open(path, 'r') as f:
        allPaperData.paperDict = json.load(f)

    if method == 'tf-idf' or method == 'bow':
        path = "dataserver/scicite/" + size + "/labeledAbst.json"
        with open(path, 'r') as f:
            labeledAbstDict = json.load(f)
        # タイトルもlabeledAbstDictへ入れておく
        for paperId in labeledAbstDict:
            labeledAbstDict[paperId]["title"] = allPaperData.paperDict[paperId]["title"]

    else:
        path = "dataserver/scicite/" + size + "/embLabel/labeledAbst" + method + ".json"
        with open(path, 'r') as f:
            labeledAbstDict = json.load(f)

    """
    データの整形
    """
    labelList = ['title', 'bg', 'obj', 'method', 'res']

    for paperId, paper in allPaperData.paperDict.items():
        # Vectorizerに合うようにアブストラクトのみをリストに抽出
        allPaperData.abstList.append(paper["abstract"])

        # 分類されたアブストラクトごとにリストに抽出
        labelAbst = labeledAbstDict[paperId]
        for label in labelList:
            if label == 'title' and method == 'tf-idf':
                allPaperData.labelList[label].append(paper["title"])
            else:
                allPaperData.labelList[label].append(labelAbst[label])

    # 辞書をリストに変換
    allPaperData.paperList = list(allPaperData.paperDict.values())

    """
    類似度測定
    """
    labelCount = {label: 0 for label in labelList}
    maxSimLabelCount = {label: 0 for label in labelList}
    allDataCount = 0
    citationIntentLabel = {}
    correct = 0
    error = 0
    intentCount = {
        'background': 0,
        'method': 0,
        'result': 0
    }
    trueMalti = []
    predMalti = []
    randomMalti = [[],[],[],[],[],[],[],[],[],[]]
    labelToNum = {
        'background': 0,
        'bg': 0,
        'obj': 0,
        'method': 1,
        'result': 2,
        'res' : 2
    }

    # Tf-IDFとBERT系で数がずれる原因究明のため
    dataList = []

    # TF-IDFの計算のためのデータのfit
    if method == 'tf-idf':
        vectorizer = TfidfVectorizer()
        vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])
    
    for data in mainData.values():
        # ターゲット論文か引用論文のタイトル/アブストが取得できていない場合は比較ができないからスキップ
        if data['citingTitle'] == None or data['citingAbstract'] == None \
            or data['citedTitle'] == None or data['citedAbstract'] == None:
            continue
        
        targetPaperId = data['citingPaperId']
        citedPaperId = data['citedPaperId']
        simDict = {}
        # 各観点毎の類似度計算
        for label in labelList:
            # タイトルは引用意図に関係ないため
            if label == 'title': continue
            
            targetLabeledAbst = labeledAbstDict[targetPaperId][label]
            citedLabeledAbst = labeledAbstDict[citedPaperId][label]
            
            # ターゲット論文か相手論文にその観点に分類されたアブスト文が無いならスキップ
            if not targetLabeledAbst or not citedLabeledAbst:
                simDict[label] = None
                continue
            
            labelCount[label] += 1
            if method == 'tf-idf':
                paperVector = vectorizer.transform([targetLabeledAbst]).toarray().tolist()[0]
                opponentPaperVector = vectorizer.transform([citedLabeledAbst]).toarray().tolist()[0]
            else:
                paperVector = targetLabeledAbst
                opponentPaperVector = citedLabeledAbst
                
            sim = cosine_similarity([paperVector], [opponentPaperVector])[0][0]
            simDict[label] = sim
        
        maxLabel, maxScore = calcMax(simDict)
        if maxLabel != None and existsCorrect(simDict, data["label"]) and existsTwoOrMoreChoice(simDict):
            judgement = judgeIntentMatching(maxLabel, data["label"])
            if judgement:
                correct += 1
            else:
                error += 1
            
            allDataCount += 1
            maxSimLabelCount[maxLabel] += 1
            trueMalti.append(labelToNum[data["label"]])
            predMalti.append(labelToNum[maxLabel])
            for i in range(len(randomMalti)):
                randomMalti[i].append(genRandomPred(simDict))
            intentCount[data["label"]] += 1
        
            dataList.append(data["unique_id"])
    
    # Tf-IDFとBERT系で数がずれる原因究明のため
    with open("result/tmpScicite-{}.json".format(method), 'w') as f:
        json.dump(dataList, f, indent=4)

    print("count", allDataCount)
    print('labelCount:', labelCount)
    print('maxCosSimLabelCount:', maxSimLabelCount)
    print("correct", correct)
    print("error", error)
    cm = confusion_matrix(trueMalti, predMalti)
    print(cm)
    print(classification_report(trueMalti, predMalti))
    randomAccuracy = []
    print("--- Random Accuracy ---")
    for i in range(len(randomMalti)):
        tmp = accuracy_score(trueMalti, randomMalti[i])
        randomAccuracy.append(tmp)
    import statistics
    print(randomAccuracy)
    print("Mean: ", statistics.mean(randomAccuracy))
    
    # 混同行列のヒートマップを画像で出力
    # plt.rcParams["font.size"] = 20
    cm = pd.DataFrame(data=cm, index=["bg", "method", "result"], 
                           columns=["bg", "method", "result"])
    sns.heatmap(cm, square=True, cbar=False, annot=True, cmap='Blues', fmt='d')
    plt.title(method)
    plt.yticks(rotation=0)
    plt.xlabel("予測", fontsize=13, rotation=0)
    plt.ylabel("正解", fontsize=13)
    plt.savefig('image/analyzeWithScicite/analyzeSciciteCM-{}.png'.format(method))

"""
データ構造の定義
"""
def calcMax(simDict):
    maxLabel = None
    maxScore = 0
    for key, value in simDict.items():
        if value and value > maxScore:
            maxScore = value
            maxLabel = key
    
    if maxLabel:
        return maxLabel, maxScore
    else:
        return None, None

def judgeIntentMatching(label, intent):
    if label == intent:
        return True
    elif label == 'bg' or label == 'obj':
        if intent == 'background':
            return True
    elif label == 'res' and intent == 'result':
        return True
    
    return False

def existsCorrect(simDict, intent):
    if intent == 'background':
        if simDict['bg'] != None or simDict['obj'] != None:
            return True
    elif intent == 'method':
        if simDict[intent] != None:
            return True
    elif intent == 'result':
        if simDict['res'] != None:
            return True
        
    return False

def existsTwoOrMoreChoice(simDict):
    # print(simDict)
    count = 0
    if simDict["bg"] != None or simDict["obj"] != None:
        count += 1
    if simDict["method"] != None:
        count += 1
    if simDict["res"] != None:
        count +=1
    
    if count >=2:
        return True
    else:
        return False
            

def genRandomPred(simDict):
    labelList = []
    if simDict["bg"] != None or simDict["obj"] != None:
        labelList.append(0)
    if simDict["method"] != None:
        labelList.append(1)
    if simDict["res"] != None:
        labelList.append(2)
        
    randomInt = random.randint(0,len(labelList)-1)
    # print(randomInt)
    return randomInt
    

if __name__ == "__main__":
    main()