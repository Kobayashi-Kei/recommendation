import json
import traceback
import lineNotifier
from transformers import *
import os

"""
BERT埋め込みなしで、タイトル、アブストラクト、分類されたアブストラクト
および引用関係のみ含むデータを整形して出力する
"""

"""
データセットのパスなどを代入
"""
#dataPath = 'data/axcell/axcellRecomData.json'
#labeledAbstPath = './data/axcell/axcellRecomData-SSCResult.txt'
dataPath = 'data/axcell/small/axcellRecomData-old.json'
labeledAbstPath = './data/axcell/small/axcellRecomData-SSCResult.txt'
outputPath = dataPath.replace('.json', '-includeLabeled.json')

"""
データのロード・整形
"""
# データセットをロード
with open(dataPath, 'r') as f:
    paperList = json.load(f)

# アブストラクトの分類結果をロード
with open(labeledAbstPath, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        tmp = json.loads(line)
        labeledAbstract = tmp[1]
        paperList[i]['labeledAbstract'] = labeledAbstract

# アブストラクトの分類結果をラベルごとに分けて連結
for i, paper in enumerate(paperList):
    texts = {
        'backgroundText': '',
        'objectiveText': '',
        'methodText': '',
        'resultText': '',
        'otherText': ''
    }

    for text_label in paper['labeledAbstract']:
        if text_label[1] == 'background_label' :
            texts['backgroundText'] += text_label[0] + ' '
        elif text_label[1] == 'objective_label':
            texts['objectiveText'] += text_label[0] + ' '
        elif text_label[1] == 'method_label':
            texts['methodText'] += text_label[0] + ' '
        elif text_label[1] == 'result_label':
            texts['resultText'] += text_label[0] + ' '
        elif text_label[1] == 'other_label':
            texts['otherText'] += text_label[0] + ' '
    
    for key in texts:
        if len(texts[key]) > 0:
            texts[key] = texts[key][:-1]

    paperList[i]['labeledAbstract'] = texts

"""
ファイル出力
"""
with open(outputPath, 'w') as f:
    json.dump(paperList, f, indent=4)
    
message = "【完了】shapeLabeledAbstract.py"
lineNotifier.line_notify(message)
