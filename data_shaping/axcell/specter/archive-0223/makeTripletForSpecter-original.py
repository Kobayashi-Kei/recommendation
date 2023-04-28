import json
import os
import random
random.seed(314)
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(314)
import sys
import pathlib
parent_dir = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
sys.path.append(parent_dir)
from tools import arg_parse_from_commandline

def main():
    """
    SPECTERの学習用にトリプレットのデータを構築する
    以下の連想配列のリスト
    {
        source: "", # ターゲット論文
        pos: "", # 被引用論文
        neg: "", # 無関係の論文
    }
    
    1つのターゲット論文に対して
    ターゲット論文 - 引用論文A  - 類似しているが引用関係に無い論文
                            - 類似しているが引用関係に無い論文
                            - ランダムな論文
                            - ランダムな論文
                            - ランダムな論文
                - 引用論文B  ...
    """
    # 引数を読み込み
    args = arg_parse_from_commandline(['method', 'titleOrAbst'])
    method = args.method

    if method != 'tf-idf' and \
        method != 'Specter':
        print("Methodの引数が間違っています")
        exit()
        
    titleOrAbst = args.titleOrAbst
    
    # サイズ
    size = "large"

    # 入力
    dataDir = "dataserver/axcell/" + size + "/"
    path = dataDir + "paperDict.json"
    with open(path, 'r') as f:
        paperDict = json.load(f)
    path = dataDir + "labeledAbst.json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)
    # 処理しやすいようにタイトルも加える
    for key in labeledAbstDict:
        labeledAbstDict[key]["title"] = key
        
    taskDataDir = dataDir + "specter/" + titleOrAbst + "-" + method + "/"
    path = taskDataDir + "taskData.json"
    with open(path, 'r') as f:
        taskData = json.load(f)
        
    # 出力パス
    outputPathPrefix = taskDataDir + "triple"
    
    labelList = ['title', 'bg', 'obj', 'method', 'res']

    tripletDataDict = {key: [] for key in labelList}

    # 負例選択用のキーのリスト
    labeledAbstDictKey = list(labeledAbstDict.keys())    

    for title, data in taskData.items():        
        targetLabeledAbst = labeledAbstDict[title]
        for label, value in targetLabeledAbst.items():
            # otherは使わない
            if label == "other":continue
            # ターゲット論文(source)にその観点の文が無いならcontinue
            if value == "":
                continue
            # labelListに無いものは飛ばす
            if not label in labelList:
                continue
            
            for citationTitle in data["citation"]:
                # 引用論文(pos)にその観点の文が無いならcontinue
                if labeledAbstDict[citationTitle][label] == "":
                    continue
                
                count = 0
                for i in range(len(data["similarNotCited"])):
                    randIdx = random.randint(0, len(data["similarNotCited"])-1)
                    # print(randIdx)
                    similarPaperTitle = data["similarNotCited"][randIdx]
                    # similarPaperにその観点の文が無いならcontinue
                    if labeledAbstDict[similarPaperTitle][label] == "":
                        continue
                    # ここまで来れたらHard Negativeなデータを生成
                    datum = {
                        "source": title,
                        "pos": citationTitle,
                        "neg": similarPaperTitle
                    }
                    tripletDataDict[label].append(datum)
                    count += 1
                    if count == 1: 
                        break
                
                # 1つの引用論文(citationData)につき1件作る
                count = 0
                while True:
                    randKey = labeledAbstDictKey[random.randint(0,len(labeledAbstDictKey)-1)]
                    if randKey == title or \
                        randKey in data["citation"] or \
                        randKey in data["similarNotCited"]:
                        continue
                    if labeledAbstDict[randKey][label] == "":
                        continue
                    
                    datum = {
                        "source": title,
                        "pos": citationTitle,
                        "neg": randKey
                    }
                    tripletDataDict[label].append(datum)
                    count += 1
                    if count == 1:
                        break
    
    # 出力
    dataClassList = ["train", "dev", "test"]
    for key in tripletDataDict:
        trainList, devTestList = train_test_split(tripletDataDict[key], test_size=0.2, shuffle=True, random_state=314)
        devList, testList =  train_test_split(devTestList, test_size=0.5, shuffle=True, random_state=314)
        
        for dataClass in dataClassList:
            outputPath = outputPathPrefix + "-" + key + "-" + dataClass + ".json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))
                
        # 統計情報の取得
        previousDatum = {}
        sourcePaperNum = 0
        posPaperNumList = []
        for datum in tripletDataDict[key]:
            if datum != previousDatum:
                sourcePaperNum += 1
                posPaperNumList.append(posPaperNumList)
                posPaperNum = 0
            else:
                posPaperNum += 1
        

if __name__ == '__main__':
    main()