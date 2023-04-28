import json
import traceback
import lineNotifier
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import os

"""
SciCiteのデータのアブストラクトの分類結果を整形・埋め込みをする
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
    #アブストラクトの分類結果の行数をキー、タイトルをバリューとする辞書
    with open(labeledLineToPaperIdPath, 'r') as f:
        labeledLineToPaperId = json.load(f)

    # アブストラクトの分類結果をロード
    labeledAbst = {}
    with open(labeledAbstPath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = json.loads(line)
            labeledAbst[labeledLineToPaperId[str(i)]] = tmp[1]
            
    # 論文データをロード
    with open(paperDictPath, 'r') as f:
        paperDict = json.load(f)
            
    # アブストラクトの分類結果をラベルごとに分けて連結
    concatLabeledAbst = {}
    for title in labeledAbst:
        texts = {
            'bg': '',
            'obj': '',
            'method': '',
            'res': '',
            'other': ''
        }

        for text_label in labeledAbst[title]:
            if text_label[1] == 'background_label' :
                texts['bg'] += text_label[0] + ' '
            elif text_label[1] == 'objective_label':
                texts['obj'] += text_label[0] + ' '
            elif text_label[1] == 'method_label':
                texts['method'] += text_label[0] + ' '
            elif text_label[1] == 'result_label':
                texts['res'] += text_label[0] + ' '
            elif text_label[1] == 'other_label':
                texts['other'] += text_label[0] + ' '
        
        for key in texts:
            if len(texts[key]) > 0:
                texts[key] = texts[key][:-1]

        concatLabeledAbst[title] = texts
    
    with open(outPutLabeledAbstPath, 'w') as f:
        json.dump(concatLabeledAbst, f, indent=4)
        
    
    """
    BERT類でラベルごとに分割した文章を埋め込む
    """

    try:
        """
        1. BERT
        """
        # モデルの初期化
        modelName  = "Bert"
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDir + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
        
        outputEmb = embedingEntire(tokenizer, model, paperDict)
        with open(outputEmbDir + "titleAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmb, f, indent=4)
        
            
        """
        2. SciBERT
        """
        # モデルの初期化
        modelName  = "SciBert"
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDir + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
        
        outputEmb = embedingEntire(tokenizer, model, paperDict)
        with open(outputEmbDir + "titleAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmb, f, indent=4)

        """
        3. SPECTER
        """
        # モデルの初期化
        modelName  = "Specter"
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter')
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDir + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
        
        outputEmb = embedingEntire(tokenizer, model, paperDict)
        with open(outputEmbDir + "titleAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmb, f, indent=4)
         
        message = "【完了】shape-and-emmbedding.py"
        lineNotifier.line_notify(message)

            
    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)

def embedingLabel(tokenizer, model, paperDict, concatLabeledAbst):
    # モデルの初期化
    model.cuda()
    model.eval()

    outputEmbLabel = {}
    
    # 埋め込み
    for title in concatLabeledAbst:
        outputEmbLabel[title] = {}
        
        # title
        input = tokenizer(paperDict[title]["title"], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
        output = model(**input).last_hidden_state[:, 0, :]
        outputEmbLabel[title]["title"] = output[0].tolist()

        # bg, obj, method, result
        # 文の間には[SEP]を挿入しない
        for key in concatLabeledAbst[title]:
            if concatLabeledAbst[title][key]:
                
                input = tokenizer(concatLabeledAbst[title][key], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
                output = model(**input).last_hidden_state[:, 0, :]
                outputEmbLabel[title][key] = output[0].tolist()
            else:
                outputEmbLabel[title][key] = None 
        
    del tokenizer
    del model
    
    return outputEmbLabel

def embedingEntire(tokenizer, model, paperDict):
    # モデルの初期化
    model.cuda()
    model.eval()

    outputEmb = {}
    
    # 埋め込み
    for title in paperDict:
        if paperDict[title]["title"] and paperDict[title]["abstract"]:
            # 文の間には[SEP]を挿入しない
            inputText = paperDict[title]["title"] + tokenizer.sep_token + paperDict[title]['abstract']
            input = tokenizer(inputText, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            outputEmb[title] = output[0].tolist()

    del tokenizer
    del model
    
    return outputEmb
        
if __name__ == "__main__":
    main()