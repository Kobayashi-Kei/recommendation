import glob
import json
import lineNotifier
"""
S2ORCデータセットから以下の条件の論文メタデータを抽出する
1. CS分野
2. アブストラクトを含む
3. 引用が1つ以上
"""
dirName = '/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/metadata/' 
extention = '*' 
outputFilePath = '/home/kobayashi/paper-recom/axcellExp/data/s2orc-cs-metadata.json'

pathlist = glob.glob(dirName + extention)
pathlist.sort()

paperList = []
countCSPaper = 0
countNotHasAbstract = 0
countNotHasCitation = 0

for path in pathlist:
    print(path)
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        paper = json.loads(line)
        
        if paper['mag_field_of_study'] == None or not ['Computer Science'] == paper['mag_field_of_study']:
            continue    
        countCSPaper += 1
        
        if paper['abstract'] == None:
            countNotHasAbstract += 1
            continue
        
        if paper['has_outbound_citations'] == False:
            countNotHasCitation += 1
            continue
        
        paperList.append(paper)

print('---- Result ----')
print('countCSPaper :', countCSPaper)
print('countNotHasAbstract :', countNotHasAbstract)
print('countNotHasCitation :', countNotHasCitation)

lineNotifier.line_notify("S2ORCのCS分野メタデータ収集完了(getCSMetadata.py)")


with open(outputFilePath, 'w') as f:
    json.dump(paperList, f, indent=4)