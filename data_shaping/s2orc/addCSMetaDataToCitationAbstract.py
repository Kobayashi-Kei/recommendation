import glob
import json
import lineNotifier
"""
S2ORCデータセットから抽出した以下の条件の論文メタデータに対して、
その引用文献のタイトルとアブストラクトを取得する
1. CS分野
2. アブストラクトを含む
3. 引用が1つ以上
"""
dataPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/s2orc-cs-metadata.json'

batch_paper_id_to_title_abstract_path = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_paper_id_to_title_abstract/'
extention = '*'
batchPathlist = glob.glob(batch_paper_id_to_title_abstract_path + extention)
batchPathlist.sort()

indexPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/paper_id_to_batch_id.json'

outputPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/s2orc-cs-metadata-includecCitationMetadata.json'

with open(dataPath, 'r') as f:
    paperList = json.load(f)
    
with open(indexPath, 'r') as f:
    paperIndexList = json.load(f)
    
for i, paper in enumerate(paperList):
    paper['citationPaper'] = []
    
    # 引用文献のループを作成
    for citation_paper_id in paper['outbound_citations']:
        batchId = paperIndexList[citation_paper_id]
        with open('/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_paper_id_to_title_abstract/paper_id_to_title_abstract_' + batchId + '.json', 'r') as f:
            titleAbstractPaperList = json.load(f)
            citationPaper = titleAbstractPaperList[citation_paper_id]
        paper['citationPaper'].append(citationPaper)

    paperList[i] = paper
    
"""
print('---- Result ----')
print('countCSPaper :', )
"""

lineNotifier.line_notify("S2ORCのCS分野の引用文献メタデータ収集完了(addCSMetaDataToCitationAbstract.py")


with open(outputPath, 'w') as f:
    json.dump(paperList, f, indent=4)