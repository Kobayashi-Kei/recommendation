import json
import statistics
import collections
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

dataPath = "dataserver/axcell/medium/paperDict.json"

# データセットをロード
with open(dataPath, 'r') as f:
    paperDict = json.load(f)
    
# テスト用のクエリ論文のタイトル
testPaperList = []
for title, paper in paperDict.items():
    if paper['test'] == 1:
        testPaperList.append(title) 

# 引用数をリストにする
numCitationList = []
citedPaperList = []
for title in testPaperList:
    numCitationList.append(len(paperDict[title]['cite']))
    for citedPaper in paperDict[title]['cite']:
        citedPaperList.append(citedPaper)
#print(numCitationList)

# 引用の引用
numCitationCitationList = []
for title in citedPaperList:
    numCitationCitationList.append(len(paperDict[title]['cite']))


print(f"ターゲット論文数: {len(numCitationList)}")
print(f"候補論文数: {len(list(paperDict.keys()))}")
# 平均・分散
mean = statistics.mean(numCitationList)
print("-ターゲット論文の引用論文数-")
print('平均: ', mean)
median = statistics.median(numCitationList)
print('中央値: ', median)
pvariance = statistics.pstdev(numCitationList)
print('分散: ', pvariance)
print("-ターゲット論文の引用論文の引用論文数")
mean = statistics.mean(numCitationCitationList)
print('平均: ', mean)
median = statistics.median(numCitationCitationList)
print('中央値: ', median)
pvariance = statistics.pstdev(numCitationCitationList)
print('標準偏差: ', pvariance)


# 全要素を数え上げて棒グラフで出力
c = collections.Counter(numCitationList)
c = sorted(c.items())
left = []
height = []
for k, v in c:
    left.append(k)
    height.append(v)

fig = plt.figure()
plt.bar(np.array(left), np.array(height))
plt.title("ターゲット論文の引用文献数での集計")
plt.xlabel("取得できた引用文献数")
plt.ylabel("ターゲット論文数")
plt.xticks(list(range(1,left[-1]+1)))
fig.savefig("image/dataset_citation.png")