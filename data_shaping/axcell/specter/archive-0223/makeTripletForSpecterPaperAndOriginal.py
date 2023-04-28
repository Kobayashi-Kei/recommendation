import json
import os
import random
random.seed(314)
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(314)

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

    
# 出力パス
outputDirPath = dataDir + size + "/specter/specterAnd" + method + "/" 
outputPathPrefix = outputDirPath + "triple"
if not os.path.exists(outputDirPath):
    os.mkdir(outputDirPath)

def main():    
    labelList = ['title', 'bg', 'obj', 'method', 'res']
    # labelList = ['bg', 'obj', 'method', 'res']
    # labelList = ["title"]
    tripletDataDict = {key: [] for key in labelList}

    # 負例選択用のキーのリスト
    labeledAbstDictKey = list(labeledAbstDict.keys())
    
    sourceCount = 0 

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
            
            for citationTitle, citationOfCitations in data.items():
                # 引用論文(pos)にその観点の文が無いならcontinue
                if labeledAbstDict[citationTitle][label] == "":
                    continue
                
                # 引用論文(pos)に引用論文が無いならcontinue
                if paperDict[citationTitle]["test"] != 1:
                    continue
                
                # hardNegativeCount = 0
                sourceCount += 1
              
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
                    tripletDataDict[label].append(datum)
                
                # 独自手法のHardNegativeを2件生成
                count = 0
                for i in range(len(myMethodData[title]["similarNotCited"])):
                    randIdx = random.randint(0, len(myMethodData[title]["similarNotCited"])-1)
                    # print(randIdx)
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
                    tripletDataDict[label].append(datum)
                    count += 1
                    if count == 2: 
                        break
                
                # 1つの引用論文(citationData)につき3件作る
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
                    tripletDataDict[label].append(datum)
                    count += 1
                    if count == 3:
                        break
                
    print("sourceCount", sourceCount)
    # print(tripletDataDict)
    
    # 出力
    dataClassList = ["train", "dev", "test"]
    for key in tripletDataDict:
        trainList, devTestList = train_test_split(tripletDataDict[key], test_size=0.2, shuffle=True, random_state=314)
        devList, testList =  train_test_split(devTestList, test_size=0.5, shuffle=True, random_state=314)
        
        for dataClass in dataClassList:
            outputPath = outputPathPrefix + "-" + key + "-" + dataClass + ".json"
            with open(outputPath, 'w') as f:
                exec("json.dump({}List, f, indent=4)".format(dataClass))

if __name__ == '__main__':
    main()