import json



embPath = 'data/axcell/small/axcellRecomData-old-includeEmb-includeLabelEmb.json'
#embPath = 'data/axcell/mini/axcellRecomData-old-includeEmb-includeLabelEmb.json'

#labeledAbstPath = "data/axcell/mini/axcellRecomData-old-includeLabeled.json"
#labeledAbstPath = "data/axcell/axcellRecomData-includeLabeled.json"

if 'embPath' in locals():
    # アブストラクト毎のemmbeddingをロード
    with open(embPath, 'r') as f:
        allEmbData = json.load(f)


    labels = ['title', 'backgroundText', 'objectiveText', 'methodText', 'resultText', 'otherText']
    shortLabels = ['title', 'background', 'objective', 'method', 'result', 'other']

    embList = []
    labelList = []
    embKey = "labeled_abstract_scibert_embedding"
    for data in allEmbData:
        for key in data[embKey]:
            if data[embKey][key] == None:
                continue
            embList.append(data[embKey][key])
            labelList.append(labels.index(key))
else:    
    # アブストラクト毎のemmbeddingをロード
    with open(labeledAbstPath, 'r') as f:
        allData = json.load(f)


    labels = ['title', 'backgroundText', 'objectiveText', 'methodText', 'resultText', 'otherText']
    shortLabels = ['title', 'background', 'objective', 'method', 'result', 'other']

    embList = []
    labelList = []
    embKey = "labeledAbstract"
    for data in allData:
        for key in data[embKey]:
            if data[embKey][key] == None:
                continue
            embList.append(data[embKey][key])
            labelList.append(labels.index(key))
        embList.append(data['title'])
        labelList.append(labels.index('title'))
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    # モデルの作成
    vectorizer = TfidfVectorizer()
    # 全体の語彙の取得とTF-IDFの計算の実行、返り値はScipyのオブジェクトとなる
    embList = vectorizer.fit_transform(embList)
    
    

#print(embList)
#exit()
# t-SNEの適用
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2) # n_componentsは低次元データの次元数
X_tsne = tsne.fit_transform(embList)

# 結果の可視化
import matplotlib.pyplot as plt
# タイトル, 背景, 目的, 手法, 結果, その他の順で色を割り当て
#colors = ['red', 'blue', 'green', 'pink', , 'purple', 'black', 'olive', 'lightblue', 'lime']
colors = ['red', 'blue', 'green', 'pink', 'tomato', 'purple']

fig = plt.figure()
plt.xlim(X_tsne[:, 0].min(), X_tsne[:, 0].max() + 1)
plt.ylim(X_tsne[:, 1].min(), X_tsne[:, 1].max() + 1)
for i in range(len(X_tsne)):
    plt.text(
        X_tsne[i, 0],
        X_tsne[i, 1],
        str(shortLabels[labelList[i]]),
        color = colors[labelList[i]]
        )
plt.xlabel('t-SNE Feature1')
plt.ylabel('t-SNE Feature2')

fig.savefig("image/visualLabel"+ embKey +".png")