import glob
import json
import lineNotifier
"""
S2ORCデータセットから各バッチの全ての論文に対して、以下の形状の辞書にしてファイル出力する
title => {abstract, paper_id} 
"""
for BATCH_ID in range(0,100):

    # filter papers using metadata values
    paperDict = {}
    with open(f'/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            title = metadata_dict['title'].strip().lower()
                
            paperDict[title] = metadata_dict        

    outputFilePath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/batch_paper_title_to_metadata/paper_title_to_metadata_' + str(BATCH_ID) + '.json'
    with open(outputFilePath, 'w') as f:
        json.dump(paperDict, f, indent=4)
        

message = "【完了】axcellExp/data_shaping/s2orc/collect_title_to_abstract.py"
lineNotifier.line_notify(message)