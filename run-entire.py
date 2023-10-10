import json

from tools import arg_parse_from_commandline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from recom import allPaperDataClass, testPaperDataClass, PaperDataClass
from recom import extractTestPaperEmbeddings, genQrels, genRun
from ranx import evaluate, compare
import numpy as np
import datetime

"""
関連研究推薦の実験
・アブストラクト全体でコサイン類似度を測り、高い順に並べて推薦順位とする
・ランキングのメトリクスで評価する
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
            method != 'bm25' and \
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
    size = "medium"
    size = 'medium-pretrain_average_pooling_entire'

    path = "dataserver/axcell/" + size + "/paperDict.json"
    with open(path, 'r') as f:
        allPaperData.paperDict = json.load(f)

    if method == 'tf-idf' or method == 'bow':
        # ラベル毎のアブストをロード
        path = "dataserver/axcell/" + size + "/labeledAbst.json"
        with open(path, 'r') as f:
            labeledAbstDict = json.load(f)
        # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
        for title in labeledAbstDict:
            labeledAbstDict[title]["title"] = title
    # else:
        # アブスト全体の埋め込みをロード
        # タイトルを用いず、アブストのみを用いる場合
        # path = "dataserver/axcell/" + size + "/embedding/titleAbst" + method + ".json"
        # with open(path, 'r') as f:
        #     abstEmb = json.load(f)


        # ラベル毎のアブスト埋め込みをロード
        # path = "dataserver/axcell/" + size + "/embLabel/labeledAbst" + method + ".json"
        # with open(path, 'r') as f:
        #     labeledAbstDict = json.load(f)

    # タイトルアブスト埋込をロード
    path = "dataserver/axcell/" + size + "/embedding/titleAbst" + method + ".json"
    with open(path, 'r') as f:
        abstEmb = json.load(f)
    """
    データの整形
    """
    labelList = ['title', 'bg', 'obj', 'method', 'res']
    for title, paper in allPaperData.paperDict.items():
        # Vectorizerに合うようにアブストラクトのみをリストに抽出
        allPaperData.abstList.append(paper["abstract"])

        if method == 'tf-idf':
            # 分類されたアブストラクトごとにリストに抽出
            labelAbst = labeledAbstDict[paper["title"]]
            for label in labelList:
                allPaperData.labelList[label].append(labelAbst[label])

    # 辞書をリストに変換
    allPaperData.paperList = list(allPaperData.paperDict.values())

    # Bert系の場合は埋め込みをリストに変換
    if 'Bert' in method or method == 'Specter':
        for title, embedAbst in abstEmb.items():
            allPaperData.abstEmbList.append(embedAbst)

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
        simMatrix = calcSimMatrix(
            allPaperData, testPaperData, vectorizer=vectorizer)

    # BOW
    elif method == 'bow':
        vectorizer = CountVectorizer()
        simMatrix = calcSimMatrix(
            allPaperData, testPaperData, vectorizer=vectorizer)

    # BERT系
    elif 'Bert' in method or 'Specter' in method:
        simMatrix = calcSimMatrix(allPaperData, testPaperData)

    # print(len(simMatrix[0]))
    # print(np.argsort(-simMatrix))
    """
    デバッグ出力
    """
    """
    print('===ラベルごとの類似度===')
    for i in range(len(simMatrix['backgroundList'])):
        print('===', i , '===')
        for j in range(len(simMatrix['backgroundList'][i])):
            print('--', j ,'--')
            print(simMatrix[i][j])
            for key in simMatrix:
                print(key, ':', simMatrix[key][i][j])
    """

    """
    分析用に正解の引用論文がモデルの予測結果では何位になっているのか出力する
    """
    outputResult = []
    for i, row in enumerate(simMatrix):
        tmpResult = {}
        tmpResult['queryTitle'] = testPaperData.paperList[i]['title']
        tmpResult['queryAbstract'] = testPaperData.paperList[i]['abstract']
        row_dict = {i: row[i] for i in range(0, len(row))}
        # print(row_dict)
        row_dict = dict(
            sorted(row_dict.items(), key=lambda x: x[1], reverse=True))
        row_rankedIndexList = list(row_dict.keys())
        # print(row_rankedIndexList)
        tmpList = []
        for citeTitle in testPaperData.paperList[i]['cite']:
            tmpDict = {}
            tmpDict['rank'] = row_rankedIndexList.index(
                allPaperData.titleToIndex[citeTitle])
            tmpDict['title'] = citeTitle
            tmpDict['abstract'] = allPaperData.abstList[allPaperData.titleToIndex[citeTitle]]
            tmpList.append(tmpDict)
        tmpResult['result'] = tmpList
        outputResult.append(tmpResult)

    outputPath = 'result/' + \
        __file__.split('/')[-1] + '-' + size + "-" + \
        dt_now.strftime('%m%d%H%M') + '.json'
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
    run = genRun(allPaperData, testPaperData, simMatrix)

    # スコア計算
    score_dict = evaluate(
        qrels, run, ["mrr", "map@10", "map@20", "recall@10", "recall@20"])
    print('{:.3f}'.format(score_dict['mrr']), '{:.3f}'.format(score_dict['map@10']), '{:.3f}'.format(
        score_dict['map@20']), '{:.3f}'.format(score_dict['recall@10']), '{:.3f}'.format(score_dict['recall@20']))


"""
Class & Methods
"""


def calcSimMatrix(allPaperData: allPaperDataClass, testPaperData: PaperDataClass, vectorizer=None):
    # TF-IDFやbowの計算を行う
    tmpVectorList = []
    if vectorizer:
        # 全体の語彙の取得とTF-IDF(bow)の計算の実行、返り値はScipyのオブジェクトとなる
        # vectorizer.fit(allPaperData.abstList)
        vectorizer.fit(allPaperData.abstList + allPaperData.labelList['title'])
        for i, text in enumerate(allPaperData.abstList):
            # vector = vectorizer.transform([text]).toarray().tolist()[0]
            titleAndAbst = text + allPaperData.labelList['title'][i]
            vector = vectorizer.transform([titleAndAbst]).toarray().tolist()[0]
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
