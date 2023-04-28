import json
import pathlib
import urllib.request as libreq
import re
import time
from email.mime.text import MIMEText
from email.utils import formatdate
import xml.etree.ElementTree as ET
import lineNotifier
"""
ファイルが設置してあるディレクトリで実行すること
"""

jsonPathList = pathlib.Path("/home/kobayashi/dataserver/2021-B4/kobayashi/axcell/pair_json/").glob('*.json')
#print(jsonPathList)

abstDict = {}
notFountAbstractCount = 0

with open('successList.json', 'r') as f:
    successList = json.load(f)
with open('failedList.json', 'r') as f:
    failedList = json.load(f)

for jsonPath in jsonPathList:
    fileName = str(jsonPath).split('/')[-1]
    id = fileName[:-5]
    id = re.search("[0-9]{4}\.[0-9]{4,5}",id).group()
    #print(id)
    if id in successList or id in failedList:
        continue
    
    time.sleep(10)
    print(jsonPath)

    query_url = 'http://export.arxiv.org/api/query?search_query=id:' + id
    try:
        with libreq.urlopen(query_url) as url:
            r = url.read()
            tree = ET.fromstring(r)
            entry = tree.find("{http://www.w3.org/2005/Atom}entry")
            if not entry:
                notFountAbstractCount += 1
                failedList.append(id)
                continue
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            if not abstract:
                notFountAbstractCount += 1
                failedList.append(id)
                continue
            abstract = abstract.replace('\n', ' ')

    except Exception as e:
        print(e)
        lineNotifier.line_notify('requestArXivAbstract.pyがエラー終了')
        with open('successList.json', 'w') as f1:
            json.dump(successList, f1, indent=4)

        with open('failedList.json', 'w') as f2:
            json.dump(failedList, f2, indent=4)
        exit()
    
    if not abstract:
        notFountAbstractCount += 1
        failedList.append(id)
        continue
    
    with open(jsonPath, 'r') as f:            
        tmpDict = json.load(f)
        
    with open(jsonPath, 'w') as f:
        tmpDict['abstract'] = abstract
        json.dump(tmpDict, f, indent=4)
        
    successList.append(id)
    
  
lineNotifier.line_notify("Axcellデータのアブストラクト取得完了(requestArXivAbstract.py")

with open('successList.json', 'w') as f1:
    json.dump(successList, f1, indent=4)

with open('failedList.json', 'w') as f2:
    json.dump(failedList, f2, indent=4)
    
print("-----Result-----")
print("notFountAbstractCount: ", notFountAbstractCount)

