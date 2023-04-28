import json
import nltk

"""
以下の形式にしてファイル出力する
{
    "abstract_id": 0, 
    "sentences":[
        ~,
        ~
    ]
}
"""
# axcellExp/data_shaping/axcell/makeTaskdataFromAxcell.pyで作成したタスクデータのパスにする
filePath = './data/axcell/axcellRecomData-old.json'
fileName = filePath.split('/')[-1]
dirPath = filePath.replace(fileName, '')
outputFileName = fileName.replace('.json', '-forSSC.jsonl')


with open(filePath, 'r') as f:
    paperList = json.load(f)

for paper in paperList:
    outputDict = {"abstract_id": 0}
    outputDict['title'] = paper['title']
    
    sentences = nltk.sent_tokenize(paper[('abstract')])
    outputDict['sentences'] = sentences

    with open(dirPath+outputFileName, 'a') as f:
        json.dump(outputDict, f)
        print('', file=f)
