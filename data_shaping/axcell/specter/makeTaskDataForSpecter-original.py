import json
import sys
import pathlib
parent_dir = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
sys.path.append(parent_dir)
from tools import arg_parse_from_commandline
from recom import allPaperDataClass, testPaperDataClass
from recom import extractTestPaperEmbeddings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from lineNotifier import line_notify
from tqdm import tqdm
import math
import numpy as np
import os
import traceback

"""
SPECTERの学習用にcitationおよびcitationOfCitaionのデータを作成する
独自の負け例選択手法を用いる
・TF-IDFが近いが引用関係にない
・SPECTER埋め込みが近いが引用関係にない
など
※ 観点の有無は考慮しない

データ例
{
    "Paper A": {
        "citation":[],
        "similarNotCited":[]
    },
    "Paper B": ...
}
"""

def main():    
    """
    引数を読み込み
    """
    args = arg_parse_from_commandline(['method', 'titleOrAbst'])
    method = args.method

    if method != 'tf-idf' and \
        method != 'Specter':
        print("Methodの引数が間違っています")
        exit()
        
    titleOrAbst = args.titleOrAbst

    """
    データ構造の定義
    """
    allPaperData = allPaperDataClass()
    testPaperData = testPaperDataClass()

    """
    データファイルの読み込み
    """
    # size = "medium"
    size = "large"
    
    # 負例のために類似する論文のリストを保持する数
    threshold = 10

    dataDir = "dataserver/axcell/"
    path = dataDir + size + "/paperDict.json"
    with open(path, 'r') as f:
        allPaperData.paperDict = json.load(f)
    
    ## largeデータなら、mediumデータを関連研究推薦の評価に使うからmediumにも含まれるものを除く
    if size == "large":
        path = dataDir + "medium" + "/paperDict.json"
        with open(path, 'r') as f:
            paperDictMedium = json.load(f)

        newDict = {}
        for title, paper in allPaperData.paperDict.items():
            if not title in paperDictMedium:
                newDict[title] = paper
        
        allPaperData.paperDict = newDict

    
    # ラベル毎のアブストをロード
    path = dataDir + size + "/labeledAbst.json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)
    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title
    
    if 'Bert' in method or method == 'Specter': 
        # tf-idfの場合はcalcSimMatrixの方で調整しているが、
        # SPECTERの場合はここで読み込むファイルを変えて調整する
        if titleOrAbst == "title":
            path = dataDir + size + "/embLabel/labeledAbst" + method + ".json"
            with open(path, 'r') as f:
                labeledAbstEmbDict = json.load(f)
            embDict = {}
            for title in labeledAbstEmbDict:
                embDict[title] = labeledAbstEmbDict[title]["title"]
                
        elif titleOrAbst == "abst":
            path = dataDir + size + "/embedding/abst" + method + ".json"
            with open(path, 'r') as f:
                embDict = json.load(f)
                
        elif titleOrAbst == "titleAbst":
            path = dataDir + size + "/embedding/titleAbst" + method + ".json"
            with open(path, 'r') as f:
                embDict = json.load(f)

    # 出力パス
    specterDirPath = dataDir + size + "/specter/"
    outputDirPath = specterDirPath + titleOrAbst + "-" + method + "/"
    outputPath = outputDirPath + "taskData.json"
    if not os.path.exists(specterDirPath):
        os.mkdir(specterDirPath)
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    
    """
    データの整形
    """
    try:
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
        
        # Bert系の場合は埋め込みをリストに変換
        if 'Bert' in method or method == 'Specter':
            # for title, embedAbst in embDict.items():
            for title in allPaperData.paperDict:
                allPaperData.abstEmbList.append(embDict[title])
        
        # テスト用のクエリ論文のインデックスを抽出
        for i, paper in enumerate(allPaperData.paperList):
            if paper['test'] == 1:
                testPaperData.paperDict[paper["title"]] = paper
                testPaperData.allDataIndex.append(i)
                testPaperData.paperList.append(paper)
                allPaperData.testDataIndex.append(len(testPaperData.paperList) - 1)
            else:
                allPaperData.testDataIndex.append(None)
                
        # 予測結果の分析のため、タイトルをキーとして、indexをバリューとする辞書を生成
        for i, paper in enumerate(allPaperData.paperList):
            allPaperData.titleToIndex[paper['title']] = i
            

        """
        BOW・TF-IDFを算出
        """
        # TF-IDF
        if method == 'tf-idf': 
            vectorizer = TfidfVectorizer()
            simMatrix = calcSimMatrix(allPaperData, testPaperData, titleOrAbst, vectorizer=vectorizer)
                
        # BOW
        elif method == 'bow':
            vectorizer = CountVectorizer()
            simMatrix = calcSimMatrix(allPaperData, testPaperData, titleOrAbst, vectorizer=vectorizer)

        # BERT系
        elif 'Bert' in method or 'Specter' in method:
            simMatrix = calcSimMatrix(allPaperData, testPaperData, titleOrAbst)
        
        rankedIndexMarix = np.argsort(-simMatrix)
        
        taskData = {}
        for idx, paper in enumerate(testPaperData.paperList):
            tmpDict = {
                    "citation": [],
                    "similarNotCited": []
                }
            for citationTitle in paper["cite"]:
                if size == "large":
                    if citationTitle in paperDictMedium:
                        continue
                tmpDict["citation"].append(citationTitle)
            
            if len(tmpDict["citation"]) == 0:
                continue
            
            count = 0
            for allDataidx in rankedIndexMarix[idx]:
                similarPaperTitle = allPaperData.paperList[allDataidx]["title"]
                if not similarPaperTitle in paper["cite"] and similarPaperTitle != paper["title"]:
                    tmpDict["similarNotCited"].append(similarPaperTitle)
                    count += 1
                if count == threshold:
                    break
            
            taskData[paper["title"]] = tmpDict
        
        with open(outputPath, 'w') as f:
            json.dump(taskData, f, indent=4)
            
        line_notify(str(__file__.split('/')[-1]) + "の" + method + " " + titleOrAbst + "が終了")
        
    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)
"""
Class & Methods
"""

def calcSimMatrix(allPaperData: allPaperDataClass, testPaperData: testPaperDataClass, titleOrAbst, vectorizer=None):
    # TF-IDFやbowの計算を行う
    tmpVectorList = []
    if vectorizer:
        if titleOrAbst == "title":
            vectorizer.fit(allPaperData.labelList["title"])
            for i, text in enumerate(allPaperData.abstList):
                input = allPaperData.labelList['title'][i]
                vector = vectorizer.transform([input]).toarray().tolist()[0]
                tmpVectorList.append(vector)
                
        elif titleOrAbst == "abst":                
            vectorizer.fit(allPaperData.abstList)
            for i, text in enumerate(allPaperData.abstList):
                input = text
                vector = vectorizer.transform([input]).toarray().tolist()[0]
                tmpVectorList.append(vector)
                
        elif titleOrAbst =="titleAbst":
            vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])
            for i, text in enumerate(allPaperData.abstList):
                input = text + allPaperData.labelList['title'][i]
                vector = vectorizer.transform([input]).toarray().tolist()[0]
                tmpVectorList.append(vector)
        
    else:             
        tmpVectorList = allPaperData.abstEmbList

    # クエリ論文のTF-IDFベクトルを抽出
    testPaperVectorList = extractTestPaperEmbeddings(
            tmpVectorList, 
            testPaperData.allDataIndex
        )
    
    # TF-IDFやBOWの場合は疎行列となるため、csr_sparse_matrixに変換して速度を上げる
    if vectorizer:
        testPaperVectorList = csr_matrix(testPaperVectorList)
        tmpVectorList = csr_matrix(tmpVectorList)
        
    simMatrix = cosine_similarity(testPaperVectorList, tmpVectorList)
    
    return simMatrix       

if __name__ == "__main__":
    main()