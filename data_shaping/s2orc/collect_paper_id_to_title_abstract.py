import glob
import json
import lineNotifier
"""
S2ORCデータセットから各バッチの全ての論文に対して、以下の形状の辞書にしてファイル出力する
paper_id => {title, abstract} 
"""
for BATCH_ID in range(0,100):

    # filter papers using metadata values
    paperDict = {}
    with open(f'/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            if metadata_dict['has_pdf_parse'] == True:
                has_pdf_parsed_abstract = metadata_dict['has_pdf_parsed_abstract']
            else:
                has_pdf_parsed_abstract = False
                
            paperDict[paper_id] = {
                'title': metadata_dict['title'],
                'abstract': metadata_dict['abstract'],
                'has_pdf_parsed_abstract': has_pdf_parsed_abstract
            }

    outputFilePath = '/home/kobayashi/paper-recom/axcellExp/data/s2orc/batch/paper_id_to_title_abstract_' + str(BATCH_ID) + '.json'
    with open(outputFilePath, 'w') as f:
        json.dump(paperDict, f, indent=4)
        

message = "【完了】axcellExp/data_shaping/s2orc/collect_paper_id_to_title_abstract.py"
lineNotifier.line_notify(message)