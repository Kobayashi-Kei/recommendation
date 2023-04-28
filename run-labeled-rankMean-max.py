import json

from tools import arg_parse_from_commandline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from ranx import evaluate
from recom import allPaperDataClass, testPaperDataClass
from recom import extractTestPaperEmbeddings, genQrels, genRun
import numpy as np
import datetime
from sklearn.metrics.pairwise import euclidean_distances

"""
関連研究推薦の実験
・分類されたアブストラクトの各観点同士で類似度の算出を行い、その和をとる
"""	
def main():
    dt_now = datetime.datetime.now()
    
    """
    引数を読み込み
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
    testPaperData = testPaperDataClass()

    """
    データファイルの読み込み
    """
    # size = "medium"
    # size = "small"
    # size = "large"
    # size = "medium-bgobj"
    # size = "medium-specterPaper_1"
    # size = "medium-abst-Specter"
    # size = "medium-title-Specter-margin4"
    # size = "medium-specterAndtitle-tf-idf"
    size = "medium-specterAndtitle-tf-idf_margin3"
    # size = "medium-specterAndtitle-tf-idfAndabst-tf-idf"
    # size = "medium-titleAbst-tf-idf"
    # size = "medium-title-Specter-margin4"
    # size = "medium-title-tf-idf_margin9"
    # size = "medium-title-Specter_scibert"
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
            
    # タイトルだけ追加学習無しモデルを利用
    sizeTitle = "medium"
    path = "dataserver/axcell/" + sizeTitle + "/embLabel/labeledAbst" + method + ".json"
    with open(path, 'r') as f:
        labeledAbstDictForTitle = json.load(f)
    
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = labeledAbstDictForTitle[title]["title"]


    """
    データの整形
    """
    labelList = ['title', 'bg', 'obj', 'method', 'res']
    # labelList = ['bg', 'obj', 'method', 'res']

    for title, paper in allPaperData.paperDict.items():
        # Vectorizerに合うようにアブストラクトのみをリストに抽出
        allPaperData.abstList.append(paper["abstract"])

        # 分類されたアブストラクトごとにリストに抽出
        labelAbst = labeledAbstDict[paper["title"]]
        for label in labelList:
            allPaperData.labelList[label].append(labelAbst[label])

    # 辞書をリストに変換
    allPaperData.paperList = list(allPaperData.paperDict.values())
    
    # テスト用のクエリ論文のインデックスを抽出
    for i, paper in enumerate(allPaperData.paperList):
        if paper['test'] == 1:
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
        simMatrixDict = calcSimMatrixForLaveled(allPaperData, testPaperData, labelList, vectorizer=vectorizer)
        mergeSimMatrix = calcMergeSimMatrix(simMatrixDict, labelList)                
            
    # BOW
    elif method == 'bow':
        vectorizer = CountVectorizer()
        simMatrixDict = calcSimMatrixForLaveled(allPaperData, testPaperData, labelList, vectorizer=vectorizer)
        mergeSimMatrix = calcMergeSimMatrix(simMatrixDict, labelList)

    # BERT系
    elif 'Bert' in method or 'Specter' in method:
        simMatrixDict = calcSimMatrixForLaveled(allPaperData, testPaperData, labelList)
        mergeSimMatrix, rankMatrixDict = calcMergeSimMatrix(simMatrixDict, labelList)

    """
    デバッグ出力
    """
    """
    print('===ラベルごとの類似度===')
    for i in range(len(simMatrixDict['backgroundList'])):
        print('===', i , '===')
        for j in range(len(simMatrixDict['backgroundList'][i])):
            print('--', j ,'--')
            print(mergeSimMatrix[i][j])
            for key in simMatrixDict:
                print(key, ':', simMatrixDict[key][i][j])
    """

    """
    分析用に正解の引用論文がモデルの予測結果では何位になっているのか出力する
    """
    outputResult = []
    for i, row in enumerate(mergeSimMatrix):
        tmpResult = {}
        tmpResult['queryTitle'] = testPaperData.paperList[i]['title']
        tmpResult['queryAbstract'] = testPaperData.paperList[i]['abstract']
        row_dict = {i: row[i] for i in range(0, len(row))}
        #print(row_dict)
        row_dict = dict(sorted(row_dict.items(), key=lambda x:x[1], reverse=True))
        row_rankedIndexList = list(row_dict.keys())
        #print(row_rankedIndexList)
        tmpList = []
        for citeTitle in testPaperData.paperList[i]['cite']:
            tmpDict = {}
            tmpDict['rank'] = row_rankedIndexList.index(allPaperData.titleToIndex[citeTitle])
            tmpDict['title'] = citeTitle
            tmpDict['abstract'] = allPaperData.abstList[allPaperData.titleToIndex[citeTitle]]
            tmpDict['labelScore'] = {}
            for key in simMatrixDict:
                tmpDict['labelScore'][key] = simMatrixDict[key][i][allPaperData.titleToIndex[citeTitle]]
            tmpDict['labelScore'] = dict(sorted(tmpDict['labelScore'].items(), key=lambda x:x[1], reverse=True))

            tmpDict['labelRank'] = {}
            for key in rankMatrixDict:
                tmpDict['labelRank'][key] = str(rankMatrixDict[key][i][allPaperData.titleToIndex[citeTitle]])            
            tmpDict['labelRank'] = dict(sorted(tmpDict['labelRank'].items(), key=lambda x:x[1]))
            # print(tmpDict["labelRank"])
            tmpList.append(tmpDict)
        tmpResult['result'] = tmpList
        outputResult.append(tmpResult)
        
    # print(outputResult)

    outputPath = 'result/' + __file__.split('/')[-1] + '-' + dt_now.strftime('%Y%m%d%H%M%S') + '.json'
    with open(outputPath, 'w') as f:
        json.dump(outputResult, f, indent=4)
        

    """
    評価
    *ranxを使う場合は、類似度スコア順に並び替えたりする必要はなく
    類似度スコアのリストでOK
    """
    # データセット情報の出力
    print('------- データセット情報 -----')
    print('クエリ論文数 :', len(testPaperData.paperList))
    print('候補（全体）論文数 :', len(allPaperData.paperList))
    countCite = 0
    for paper in testPaperData.paperList:
        countCite += len(paper['cite'])
    print('クエリ論文の平均引用文献数: ', countCite/len(testPaperData.paperList))
        
    qrels = genQrels(testPaperData)
    run = genRun(allPaperData, testPaperData, mergeSimMatrix)
    
    with open("tmpRun.json" , "w") as f:
        json.dump(dict(run), f, indent=4)

    # スコア計算
    score_dict = evaluate(qrels, run, ["mrr","map@10", "map@20","recall@10","recall@20"])
    print('{:.3f}'.format(score_dict['mrr']), '{:.3f}'.format(score_dict['map@10']), '{:.3f}'.format(score_dict['map@20']), '{:.3f}'.format(score_dict['recall@10']), '{:.3f}'.format(score_dict['recall@20']))

"""
Class & Methods
"""
def calcSimMatrixForLaveled(allPaperData: allPaperDataClass, testPaperData: testPaperDataClass, labelList, vectorizer=None):
    # 背景、手法などの類似度を計算した行列を格納する辞書
    simMatrixDict =  {v : [] for v in labelList}
    
    # TF-IDFやbowの計算を行う
    if vectorizer:
        # 全体の語彙の取得とTF-IDF(bow)の計算の実行、返り値はScipyのオブジェクトとなる
        # vectorizer.fit(allPaperData.abstList)
        vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])

    for key in labelList:
        # そのラベルに分類された文章がないことで、ベクトルがNoneとなっているものを記憶しておく
        isNotNoneMatrix = np.ones((len(testPaperData.paperList), len(allPaperData.paperList)))
        if vectorizer:
            tmpVectorList = []
            # ベクトルに変換
            for i, text in enumerate(allPaperData.labelList[key]):
                
                if text:
                    # ここで行列に変換されてしまうため[0]を参照する
                    vector = vectorizer.transform([text]).toarray().tolist()[0]
                else:
                    # cosine_simをまとめて計算するために、Noneではなく0(なんでもいい)を代入しておく
                    vector = [0]*len(vectorizer.get_feature_names_out())
                    # その場合のインデックスを覚えておく
                    isNotNoneMatrix[:,i] = 0
                    if i in testPaperData.allDataIndex:
                        isNotNoneMatrix[allPaperData.testDataIndex[i],:] = 0
                tmpVectorList.append(vector)
        else:
            tmpVectorList = allPaperData.labelList[key]
            for i, vector in enumerate(tmpVectorList):
                if vector == None:
                    # cosine_simをまとめて計算するために、Noneではなく0(なんでもいい)を代入しておく
                    tmpVectorList[i] = [0]*768 # BERTの次元数
                    # その場合のインデックスを覚えておく
                    isNotNoneMatrix[:,i] = 0
                    if i in testPaperData.allDataIndex:
                        isNotNoneMatrix[allPaperData.testDataIndex[i],:] = 0

        # クエリ論文のTF-IDFベクトルを抽出
        testPaperVectorList = extractTestPaperEmbeddings(
                tmpVectorList, 
                testPaperData.allDataIndex
            )
        #simMatrixDict[key] = calcSimMatrix(testPaperVectorList, tmpVectorList)
        #print(isNoneVectorMatrix)
        # TF-IDFやBOWの場合は疎行列となるため、csr_sparse_matrixに変換して速度を上げる
        if vectorizer:
            testPaperVectorList = csr_matrix(testPaperVectorList)
            tmpVectorList = csr_matrix(tmpVectorList)
        # simMatrix = euclidean_distances(testPaperVectorList, tmpVectorList)
        simMatrix = cosine_similarity(testPaperVectorList, tmpVectorList)
        
        # 本来はテキストがなかったものをNanに変換する
        # 懸念点としては、元からcosine_simが0だったものもNanに変換されてしまうこと。
        simMatrix = simMatrix*isNotNoneMatrix
        simMatrixDict[key] = np.where(simMatrix==0, np.nan, simMatrix)

    return simMatrixDict       

def calcMergeSimMatrix(simMatrixDict, labelList):
    # simMatrixDictのラベルごとの各要素で平均を取る
    mergeRankMatrix = np.zeros((len(simMatrixDict[labelList[0]]),len(simMatrixDict[labelList[0]][0])))
    
    # 要素がnanでなければ1、nanなら0を立てた行列をラベル毎に格納した辞書
    notNanSimMatrixDict = {} 
    for key in simMatrixDict:
        notNanSimMatrixDict[key] = np.where(np.isnan(simMatrixDict[key]), 0, 1)

    # nanでない要素数の合計をもとめる
    notNanSam = sum([notNanSimMatrixDict[key] for key in notNanSimMatrixDict])

    # ndarrayの加算の都合上、simMatrixのnanを0に変換する
    for key in simMatrixDict:
        simMatrixDict[key] = np.where(np.isnan(simMatrixDict[key]), 0, simMatrixDict[key])
    
    rankMatrixDict = {}
    for key in simMatrixDict:
        rankMatrixDict[key] = np.argsort(-simMatrixDict[key]) 
        tmpMatrix = []
        for row in rankMatrixDict[key]:
            tmpList = [0 for i in range(len(row))]
            for i, idx in enumerate(row):
                tmpList[idx] = i
            tmpMatrix.append(tmpList)
        rankMatrixDict[key] = np.array(tmpMatrix) 
        rankMatrixDict[key] = rankMatrixDict[key] * notNanSimMatrixDict[key]
    
    print(rankMatrixDict)
    
    rankReverseMatrixDict = {}
    # simMatrixの0をnanに変換
    for key in rankMatrixDict:
        rankReverseMatrixDict[key] = 1/np.where(rankMatrixDict[key] == 0, np.nan, rankMatrixDict[key])
        
    print(rankMatrixDict)
    
    # 全てのrankMatrixの要素の最小を出す
    for key in rankReverseMatrixDict:
        mergeRankMatrix = np.fmax(mergeRankMatrix, rankReverseMatrixDict[key])

    print(mergeRankMatrix)

    # return mergeSimMatrix
    return mergeRankMatrix, rankMatrixDict

if __name__ == "__main__":
    main()