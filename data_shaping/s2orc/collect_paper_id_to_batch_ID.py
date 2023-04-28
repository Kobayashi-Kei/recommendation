import glob
import json
import lineNotifier
"""
S2ORCデータセットから各バッチの全ての論文に対して、以下の形状の辞書にしてファイル出力する
paper_id => バッチファイルのID
"""
paperDict = {}
for BATCH_ID in range(0,100):

    # filter papers using metadata values
    
    with open(f'/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            paperDict[paper_id] = BATCH_ID

outputFilePath = '/home/kobayashi/dataserver/2021-B4/kobayashi/s2orc/paper_id_to_batch_id.json'
with open(outputFilePath, 'w') as f:
    json.dump(paperDict, f, indent=4)
        

message = "【完了】axcellExp/data_shaping/s2orc/collect_paper_id_to_title_abstract.py"
lineNotifier.line_notify(message)