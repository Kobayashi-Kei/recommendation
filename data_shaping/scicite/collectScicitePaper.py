import re
import json
import requests
import time
import sys
import pathlib
import os
import traceback
# モジュール検索パスに，ひとつ上の階層の絶対パスを追加
# sys.path.append(os.path.abspath("../../"))
# ひとつ上の階層の絶対パスを取得
parent_dir = str(pathlib.Path(__file__).parent.parent.parent.resolve())
# モジュール検索パスに，ひとつ上の階層の絶対パスを追加
sys.path.append(parent_dir)
from lineNotifier import line_notify

def collectPaperData(originalDataPath, outputPath):
    count = 0
    
    with open(outputPath, 'r') as f:
        data = json.load(f)
        
    with open(originalDataPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datum = json.loads(line)
            #print(datum)
            
            try: 
                # すでに取得済みの場合
                if datum['citingPaperId'] in data:
                    continue
                
                urlCitingPaper = "https://api.semanticscholar.org/graph/v1/paper/" + datum['citingPaperId'] + "?fields=title,abstract"
                headers = {"content-type" : "application/json"}
                r = requests.get(urlCitingPaper, headers=headers)
                jsonData = r.json()
                if not 'error' in jsonData:
                    if 'title' in jsonData:
                        datum['citingTitle'] = jsonData['title']
                    if 'abstract' in jsonData:
                        datum['citingAbstract'] = jsonData['abstract']
                
                time.sleep(8)
            
                urlCitedPaper = "https://api.semanticscholar.org/graph/v1/paper/" + datum['citedPaperId'] + "?fields=title,abstract"
                headers = {"content-type" : "application/json"}
                r = requests.get(urlCitedPaper, headers=headers)
                jsonData = r.json()
                if not 'error' in jsonData:
                    if 'title' in jsonData:
                        datum['citedTitle'] = jsonData['title']
                    if 'abstract' in jsonData:
                        datum['citedAbstract'] = jsonData['abstract']
                
                data[datum['citingPaperId']] = datum
                count += 1
                
                time.sleep(8)
                print(count)
                
            except:
                # リクエストの上限エラーなどが発生したら、キャッチして一旦ファイル出力
                with open(outputPath, 'w') as f:
                    json.dump(data, f, indent=4)
                trace = traceback.format_exc()
                print(trace)
                line_notify(trace)
                exit()
                
    
    with open(outputPath, 'w') as f:
        json.dump(data, f, indent=4)

trainPath = "data/scicite/train.jsonl"
devPath = "data/scicite/dev.jsonl"
testPath = "data/scicite/test.jsonl"

collectPaperData(trainPath, trainPath.replace(".jsonl", "-withAbst.json"))
line_notify(str(__file__.split('/')[-1]) + "のtrainが終了")
collectPaperData(devPath, devPath.replace(".jsonl", "-withAbst.json"))
line_notify(str(__file__.split('/')[-1]) + "のdevが終了")
collectPaperData(testPath, testPath.replace(".jsonl", "-withAbst.json"))
line_notify(str(__file__.split('/')[-1]) + "のtestが終了")