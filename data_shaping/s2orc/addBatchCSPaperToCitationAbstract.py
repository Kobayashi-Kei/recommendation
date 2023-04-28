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
dataDirPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_cs_pdf_abst_bodytext_with_relatedwork_citation/'
batch_paper_title_to_metadata_path = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_paper_title_to_metadata/'
indexPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/title_to_batch_id.json'
outputDirPath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_cs_pdf_abst_bodytext_with_relatedwork_citation_include_metadata/'

with open(indexPath, 'r') as f:
    paperIndexList = json.load(f)

for BATCH_ID in range(0,100):
    with open(dataDirPath + 's2orc-cs-with-related-work-citaion_{BATCH_ID}.json', 'r') as f:
        paperList = json.load(f)
    for i, paper in enumerate(paperList):
        paper['citationPaper'] = []
        
        # 引用文献のループを作成
        for citation_paper_title in paper['citingPaperTitleListInRelatedWork']:
            citation_paper_title = citation_paper_title.strip().lower()
            if citation_paper_title in paperIndexList:
                batchId = paperIndexList[citation_paper_title]
                with open('/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_paper_id_to_title_abstract/paper_id_to_title_abstract_' + batchId + '.json', 'r') as f:
                    titleAbstractPaperList = json.load(f)
                    citationPaper = titleAbstractPaperList[citation_paper_title]
                paper['citationPaper'].append(citationPaper)

        paperList[i] = paper

    with open(outputDirPath + '{BATCH_ID}.json', 'w') as f:
        json.dump(paperList, f, indent=4)
        
"""
print('---- Result ----')
print('countCSPaper :', )
"""

lineNotifier.line_notify("S2ORCのCS分野の関連研究章の引用文献メタデータ収集完了(addBatchCSPaperToCitationAbstract.py)")
