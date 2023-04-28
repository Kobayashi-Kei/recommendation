import json
import traceback
import lineNotifier
from transformers import *
import os
from train import Specter

"""
Axcellのデータのアブストラクトの分類結果を整形・埋め込みをする
アブストラクトの分類結果を整形・埋め込みをする
"""	
def main():
    """
    データセットのパスなどを代入
    """
    datasetName = "train"
    paperDictPath =  "dataserver/scicite/"+ datasetName + "/paperDict.json"
    labeledAbstPath = "dataserver/scicite/" +  datasetName + "/sscResult.jsonl"
    labeledLineToPaperIdPath =  "dataserver/scicite/" +  datasetName + "/sscInputLineNumToPaperId.json"

    outPutLabeledAbstPath = "dataserver/scicite/" +  datasetName + "/labeledAbst.json"
    outputEmbDir = "dataserver/scicite/" +  datasetName + "/embedding/"
    outputEmbLabelDir = "dataserver/scicite/" +  datasetName + "/embLabel/"

    if not os.path.exists(outputEmbDir):
        os.mkdir(outputEmbDir)
    if not os.path.exists(outputEmbLabelDir):
        os.mkdir(outputEmbLabelDir)

    """
    データのロード・整形
    """
    # データセットをロード
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)


    # アブストラクトの分類結果をロード
    with open(labeledAbstPath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = json.loads(line)
            labeledAbstract = tmp[1]
            paperDict[i]['labeledAbstract'] = labeledAbstract

    # アブストラクトの分類結果をラベルごとに分けて連結
    for i, paper in enumerate(paperDict):
        texts = {
            'bg': '',
            'obj': '',
            'method': '',
            'res': '',
            'other': ''
        }

        for text_label in paper['labeledAbstract']:
            if text_label[1] == 'background_label' :
                texts['bg'] += text_label[0] + ' '
            elif text_label[1] == 'objective_label':
                texts['obj'] += text_label[0] + ' '
            elif text_label[1] == 'method_label':
                texts['method'] += text_label[0] + ' '
            elif text_label[1] == 'result_label':
                texts['res'] += text_label[0] + ' '
            elif text_label[1] == 'other_label':
                texts['otherText'] += text_label[0] + ' '
        
        for key in texts:
            if len(texts[key]) > 0:
                texts[key] = texts[key][:-1]

        paperDict[i]['labeledAbstract'] = texts

    """
    BERT類でラベルごとに分割した文章を埋め込む
    """

    try:
        """
        1. BERT
        """
        # モデルの初期化
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.cuda()
        model.eval()

        # 埋め込み
        for i, paper in enumerate(paperDict):
            # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
            paperDict[i]['labeled_abstract_bert_embedding'] = {}
            for key in paper['labeledAbstract']:
                if paper['labeledAbstract'][key]:
                    input = tokenizer(paper['labeledAbstract'][key], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
                    # model(**input).pooler_output にすることも検討
                    output = model(**input).last_hidden_state[:, 0, :]
                    paperDict[i]['labeled_abstract_bert_embedding'][key] = output[0].tolist()
                else:
                    paperDict[i]['labeled_abstract_bert_embedding'][key] = None
                    
            input = tokenizer(paper['title'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperDict[i]['labeled_abstract_bert_embedding']['title'] = output[0].tolist()

        del tokenizer
        del model
            
        """
        2. SciBERT
        """
        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        model.cuda()
        model.eval()

        # 埋め込み
        for i, paper in enumerate(paperDict):
            paperDict[i]['labeled_abstract_scibert_embedding'] = {}
            for key in paper['labeledAbstract']:
                if paper['labeledAbstract'][key]:
                    input = tokenizer(paper['labeledAbstract'][key], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
                    # model(**input).pooler_output にすることも検討
                    output = model(**input).last_hidden_state[:, 0, :]
                    paperDict[i]['labeled_abstract_scibert_embedding'][key] = output[0].tolist()
                else:
                    paperDict[i]['labeled_abstract_scibert_embedding'][key] = None
                    
            input = tokenizer(paper['title'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperDict[i]['labeled_abstract_scibert_embedding']['title'] = output[0].tolist()
            
        del tokenizer
        del model

        """
        3. SPECTER
        """
        # モデルの初期化
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter')
        model.cuda()
        model.eval()

        # 埋め込み
        for i, paper in enumerate(paperDict):
            paperDict[i]['labeled_abstract_specter_embedding'] = {}
            for key in paper['labeledAbstract']:
                if paper['labeledAbstract'][key]:
                    input = tokenizer(paper['labeledAbstract'][key], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
                    # model(**input).pooler_output にすることも検討
                    output = model(**input).last_hidden_state[:, 0, :]
                    paperDict[i]['labeled_abstract_specter_embedding'][key] = output[0].tolist()
                else:
                    paperDict[i]['labeled_abstract_specter_embedding'][key] = None
                    
            input = tokenizer(paper['title'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperDict[i]['labeled_abstract_specter_embedding']['title'] = output[0].tolist()
                    
        del tokenizer
        del model
        
    
        """
        ファイル出力
        """
        with open(outputPath, 'w') as f:
            json.dump(paperDict, f, indent=4)
            
        message = "【完了】shape-and-emmbedding.py"
        lineNotifier.line_notify(message)

            
    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


if __name__ == '__main__':
    main()