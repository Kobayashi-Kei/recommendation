import json
import random
import os
random.seed(314)

def main():
    """
    SPECTERの学習用にcitationおよびcitationOfCitaionのデータを作成する
    ※ 観点の有無は考慮しない
    
    データ例
    {
        "Paper A": {
            "citation":[],
            "citationOfCitation":[]
        },
        "Paper B": ...
    }
    """
    size = "large"

    # 入力
    dataDir = "dataserver/axcell/"
    path = dataDir + size + "/paperDict.json"
    with open(path, 'r') as f:
        paperDict = json.load(f)
    path = dataDir + size + "/labeledAbst.json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)

    if size == "large": # largeデータなら、mediumデータを関連研究推薦の評価に使うからmediumにも含まれるものを除く
        path = dataDir + "medium" + "/paperDict.json"
        with open(path, 'r') as f:
            paperDictMedium = json.load(f)
        
    # 出力
    outputDirPath = dataDir + size + "/specter/"
    outputPath = outputDirPath + "taskData.json"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)

    
    # labelList = ['title', 'bg', 'obj', 'method', 'res']
    taskData = {}

    for title, paper in paperDict.items():
        if paper["test"] != 1 or\
            title in paperDictMedium:
                continue
        tmpDict = {}
        
        citationOfCitationFlag = False
        for citationTitle in paper["cite"]:
            if citationTitle in paperDictMedium:
                continue
            tmpList = []
            for citationOfCitationTitle in paperDict[citationTitle]["cite"]:
                if citationOfCitationTitle in paperDictMedium:
                    continue
                tmpList.append(citationOfCitationTitle)
                citationOfCitationFlag = True
                
            print(citationTitle)
            tmpDict[citationTitle] = tmpList
        if len(list(tmpDict.keys())) > 0:
            # print(citationOfCitationFlag)
            if citationOfCitationFlag:
                taskData[title] = tmpDict
    
    with open(outputPath, 'w') as f:
        json.dump(taskData, f, indent=4)

if __name__ == '__main__':
    main()