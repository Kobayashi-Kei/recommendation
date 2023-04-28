import glob
import json

"""
Axcellのデータから関連研究推薦用のタスクデータを生成して、
json形式ファイルで出力する

"""

dirName = '/home/kobayashi/dataserver/2021-B4/kobayashi/axcell/pair_json/'
pathlist = glob.glob(dirName + '*')
pathlist.sort()

paperList = []
paperIndex = []
for path in pathlist:
    with open(path, 'r') as f:
        paperData = json.load(f)
        # タイトルかAbstractが存在しなければ飛ばす
        if not 'title' in paperData or not 'abstract' in paperData:
            continue
        
        # 引用文献が取得できていなければ飛ばす
        if len(paperData['cited_paper_title']) == 0:
            continue 
        
        # paperListに存在すれば、引用文献タイトルのリストとtestフラグだけ立てる
        if not paperData['title'] in paperIndex:
            tmpDict = {}
            tmpDict['title'] = paperData['title']
            tmpDict['abstract'] = paperData['abstract']
            tmpDict['cite'] = list(paperData['cited_paper_title'].values()) 
            tmpDict['test'] = 1
            paperList.append(tmpDict)
            paperIndex.append(tmpDict['title'])
        else:
            index = paperIndex.index(paperData['title'])
            paperList[index]['cite'] = list(paperData['cited_paper_title'].values())
            paperList[index]['test'] = 1
            
        for cited_key in paperData['cited_paper_title']:
            # 被引用文献はpaperListに存在しない場合のみ、情報を入れる
            if not paperData['cited_paper_title'][cited_key] in paperIndex:
                tmpDict = {}
                tmpDict['title'] = paperData['cited_paper_title'][cited_key]
                tmpDict['abstract'] = paperData['CitedPoolText'][cited_key].replace('\n', ' ').replace('  ',' ')
                tmpDict['cite'] = []
                tmpDict['test'] = 0
                paperList.append(tmpDict)
                paperIndex.append(tmpDict['title'])
            #else:
            #    print(paperData['cited_paper_title'][cited_key])
                
outputFilePath = './data/axcell/axcellRecomData.json'
with open(outputFilePath, 'w') as f:
    json.dump(paperList, f, indent=4)