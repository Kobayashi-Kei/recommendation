from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import random
random.seed(314)
np.random.seed(314)

size = "large"

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

path = dataDir + size + "/specter/paper/taskData.json"
with open(path, 'r') as f:
    taskData = json.load(f)

# 出力パス
outputDirPath = dataDir + size + "/specter/paper/"
outputPathPrefix = outputDirPath + "triple"
if not os.path.exists(outputDirPath):
    os.mkdir(outputDirPath)


def main():
    """
    SPECTERの学習用にトリプレットのデータを構築する
    以下の連想配列のリスト
    {
        source: "", # ターゲット論文
        pos: "", # 被引用論文
        neg: "", # 無関係の論文
    }

    1つのターゲット論文に対して source - pos - neg
    ターゲット論文 - 引用論文A  - 引用論文Aの引用論文1
                            - 引用論文Aの引用論文2
                            - ランダム論文1
                - 引用論文B  - 引用論文Bの引用論文1
                            - 引用論文Bの引用論文2
                            - ランダム論文1
    """

    tripletDataList = []

    # 負例選択用のキーのリスト
    labeledAbstDictKey = list(labeledAbstDict.keys())

    for title, data in taskData.items():
        targetLabeledAbst = labeledAbstDict[title]

        for citationTitle, citationOfCitations in data.items():
            # 引用論文(pos)に引用論文が無いならcontinue
            if paperDict[citationTitle]["test"] != 1:
                continue

            # hardNegativeCount = 0

            for citationOfCitationTitle in citationOfCitations:
                # ここまで来れたらHard Negativeなデータを生成
                datum = {
                    "source": title,
                    "pos": citationTitle,
                    "neg": citationOfCitationTitle
                }
                tripletDataList.append(datum)

            # 1つの引用論文(citationData)につき３件作る
            count = 0
            while True:
                randKey = labeledAbstDictKey[random.randint(
                    0, len(labeledAbstDictKey)-1)]
                # 引用および引用の引用に被っていたらcontinue
                if randKey == title or \
                        randKey in data or \
                        randKey in citationOfCitations:
                    continue

                datum = {
                    "source": title,
                    "pos": citationTitle,
                    "neg": randKey
                }
                tripletDataList.append(datum)
                count += 1
                # easynegative は1件
                if count == 1:
                    break

    # 出力
    dataClassList = ["train", "dev", "test"]
    trainList, devTestList = train_test_split(
        tripletDataList, test_size=0.2, shuffle=True, random_state=314)
    devList, testList = train_test_split(
        devTestList, test_size=0.5, shuffle=True, random_state=314)

    for dataClass in dataClassList:
        outputPath = outputPathPrefix + "-" + dataClass + ".json"
        with open(outputPath, 'w') as f:
            exec("json.dump({}List, f, indent=4)".format(dataClass))


if __name__ == '__main__':
    main()
