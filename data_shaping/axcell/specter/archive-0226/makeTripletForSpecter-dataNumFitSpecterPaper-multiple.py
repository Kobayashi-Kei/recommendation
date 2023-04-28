import json
import os
import random
random.seed(314)
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(314)
import statistics

size = "large"
method = "title-tf-idf"
method2 = "abst-tf-idf"

# 入力
dataDir = "dataserver/axcell/"
path = dataDir + size + "/paperDict.json"
with open(path, 'r') as f:
    paperDict = json.load(f)
path = dataDir + size + "/labeledAbst.json"
with open(path, 'r') as f:
    labeledAbstDict = json.load(f)
# 処理しやすいようにタイトルも加える
for key in labeledAbstDict:
    labeledAbstDict[key]["title"] = key

path = dataDir + size + "/specter/specterPaper/taskData.json"
with open(path, 'r') as f:
    taskData = json.load(f)

path = dataDir + size + "/specter/" + method + "/taskData.json"
with open(path, 'r') as f:
    myMethodData = json.load(f)
    
path = dataDir + size + "/specter/" + method2 + "/taskData.json"
with open(path, 'r') as f:
    myMethodData2 = json.load(f)


# 出力パス
outputDirPath = dataDir + size + "/specter/" + method + "/" 
outputPathPrefix = outputDirPath + "triple"
if not os.path.exists(outputDirPath):
    os.mkdir(outputDirPath)


def main():
    labelList = ['title', 'bg', 'obj', 'method', 'res']

    tripletDataDictOfSpecterPaper = {key: [] for key in labelList}
    tripletDataDictOfOriginal = {key: [] for key in labelList}
    tripletDataDict2 = {key: [] for key in labelList}
    tripletDataDict3 = {key: [] for key in labelList}
    

    # 負例選択用のキーのリスト
    labeledAbstDictKey = list(labeledAbstDict.keys())
    
    sourceCount = 0 
    easyNegativeCountList = []
    hardNegativeCountList = []
    negativeCountList = []

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
            # hardNegativeCount = 0
            sourceCount += 1
            hardNegativeCount = 0
            easyNegativeCount = 0
            
            for citationTitle, citationOfCitations in data.items():
                # 引用論文(pos)にその観点の文が無いならcontinue
                if labeledAbstDict[citationTitle][label] == "":
                    continue
                
                # 引用論文(pos)に引用論文が無いならcontinue
                if paperDict[citationTitle]["test"] != 1:
                    continue
              
                for citationOfCitationTitle in citationOfCitations:
                    # Citaion of Citationにその観点の文が無いならcontinue
                    if labeledAbstDict[citationOfCitationTitle][label] == "":
                        continue
                    
                    # ここまで来れたらHard Negativeなデータを生成
                    datum = {
                        "source": title,
                        "pos": citationTitle,
                        "neg": citationOfCitationTitle
                    }
                    tripletDataDictOfSpecterPaper[label].append(datum)
                    tripletDataDict2[label].append(datum)
                    tripletDataDict3[label].append(datum)
                    hardNegativeCount += 1
                
                    # 独自手法1のHardNegativeのデータを生成
                    for i in range(len(myMethodData[title]["similarNotCited"])):
                        randIdx = random.randint(0, len(myMethodData[title]["similarNotCited"])-1)
                        # print(myMethodData[title]["similarNotCited"])
                        similarPaperTitle = myMethodData[title]["similarNotCited"][randIdx]
                        # similarPaperにその観点の文が無いならcontinue
                        if labeledAbstDict[similarPaperTitle][label] == "":
                            continue
                        # ここまで来れたらHard Negativeなデータを生成
                        datum = {
                            "source": title,
                            "pos": citationTitle,
                            "neg": similarPaperTitle
                        }
                        tripletDataDictOfOriginal[label].append(datum)
                        tripletDataDict2[label].append(datum)
                        tripletDataDict3[label].append(datum)
                        break
                    
                    # 独自手法2のHardNegativeのデータを生成
                    for i in range(len(myMethodData2[title]["similarNotCited"])):
                        randIdx = random.randint(0, len(myMethodData2[title]["similarNotCited"])-1)
                        # print(myMethodData2[title]["similarNotCited"])
                        similarPaperTitle = myMethodData2[title]["similarNotCited"][randIdx]
                        # similarPaperにその観点の文が無いならcontinue
                        if labeledAbstDict[similarPaperTitle][label] == "":
                            continue
                        # ここまで来れたらHard Negativeなデータを生成
                        datum = {
                            "source": title,
                            "pos": citationTitle,
                            "neg": similarPaperTitle
                        }
                        tripletDataDict3[label].append(datum)
                        break

                
                # 1つの引用論文(citationData)につき1件作る
                count = 0
                while True:
                    randKey = labeledAbstDictKey[random.randint(0,len(labeledAbstDictKey)-1)]
                    if randKey == title or \
                        randKey in data or \
                        randKey in citationOfCitations:
                        continue
                        
                    if labeledAbstDict[randKey][label] == "":
                        continue
                    datum = {
                        "source": title,
                        "pos": citationTitle,
                        "neg": randKey
                    }
                    tripletDataDictOfSpecterPaper[label].append(datum)
                    tripletDataDictOfOriginal[label].append(datum)
                    tripletDataDict2[label].append(datum)
                    tripletDataDict3[label].append(datum)
                    easyNegativeCount += 1
                    count += 1
                    if count == 1:
                        break
            hardNegativeCountList.append(hardNegativeCount)
            easyNegativeCountList.append(easyNegativeCount)      
            negativeCountList.append(hardNegativeCount+easyNegativeCount)       
                
    print("クエリ論文数: ", sourceCount)
    print("Negative数の平均値: ", statistics.mean(negativeCountList))
    print("Negative数の中央値: ", statistics.median(negativeCountList))
    print("HardNegative数の平均値: ", statistics.mean(hardNegativeCountList))
    print("HardNegative数の中央値: ", statistics.median(hardNegativeCountList))
    print("easyNegative数の平均値: ", statistics.mean(easyNegativeCountList))
    print("easyNegative数の中央値: ", statistics.median(easyNegativeCountList))
    
    exit()

    # 出力
    dataClassList = ["train", "dev"]
    for key in tripletDataDictOfSpecterPaper:
        trainList, devList = train_test_split(tripletDataDictOfSpecterPaper[key], test_size=0.1, shuffle=True, random_state=314)    
        for dataClass in dataClassList:
            outputPath = f"{dataDir}{size}/specter/specterPaper/triple-{key}-{dataClass}.json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))
            print(outputPath)
    for key in tripletDataDictOfOriginal:
        trainList, devList = train_test_split(tripletDataDictOfOriginal[key], test_size=0.1, shuffle=True, random_state=314)    
        for dataClass in dataClassList:
            outputPath = outputPathPrefix + "-" + key + "-" + dataClass + ".json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))
            print(outputPath)
    for key in tripletDataDict2:
        trainList, devList = train_test_split(tripletDataDict2[key], test_size=0.1, shuffle=True, random_state=314)    
        for dataClass in dataClassList:
            outputPath = f"{dataDir}{size}/specter/specterAnd{method}/triple-{key}-{dataClass}.json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))
            print(outputPath)
    for key in tripletDataDict3:
        trainList, devList = train_test_split(tripletDataDict3[key], test_size=0.1, shuffle=True, random_state=314)    
        for dataClass in dataClassList:
            outputPath = f"{dataDir}{size}/specter/specterAnd{method}And{method2}/triple-{key}-{dataClass}.json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))
            print(outputPath)
                

if __name__ == '__main__':
    main()