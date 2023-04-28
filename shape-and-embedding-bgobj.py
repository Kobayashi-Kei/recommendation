import json
import traceback
import lineNotifier
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import os

"""
Axcellデータセットのアブストラクトの分類結果を整形・埋め込みをする
※ objectiveに分類された文章はbackgroundに含める
"""	
def main():
    """
    データセットのパスなどを代入
    """
    size = "medium"
    dirPath = "dataserver/axcell/" + size
    
    # 入力
    dataPath = dirPath + "/paperDict.json"
    sscResultPath = dirPath + "/resultSSC.txt"
    
    # 出力
    outputDirPath = dirPath + "-bgobj/"
    outputEmbLabelDirPath = outputDirPath + "embLabel/"
    if not os.path.exists(outputDirPath):
        os.mkdir(outputDirPath)
    if not os.path.exists(outputEmbLabelDirPath):
        os.mkdir(outputEmbLabelDirPath)

    """
    データのロード・整形
    """
    # データセットをロード
    with open(dataPath, 'r') as f:
        paperDict = json.load(f)
    
    paperDictKeys = list(paperDict.keys())    

    # アブストラクトの分類結果をロード
    with open(sscResultPath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = json.loads(line)
            labeledAbstract = tmp[1]
            paperDict[paperDictKeys[i]]['sscResult'] = labeledAbstract
    
    # アブストラクトの分類結果をラベルごとに分けて連結
    concatLabeledAbst = {}
    for title, paper in paperDict.items():
        texts = {
            'bg': '',
            'method': '',
            'res': '',
            'other': ''
        }

        for text_label in paper['sscResult']:
            if text_label[1] == 'background_label' or \
                text_label[1] == 'objective_label':
                texts['bg'] += text_label[0] + ' '
            elif text_label[1] == 'method_label':
                texts['method'] += text_label[0] + ' '
            elif text_label[1] == 'result_label':
                texts['res'] += text_label[0] + ' '
            elif text_label[1] == 'other_label':
                texts['other'] += text_label[0] + ' '
        
        for key in texts:
            if len(texts[key]) > 0:
                texts[key] = texts[key][:-1]

        paperDict[title]['labeledAbstract'] = texts
        concatLabeledAbst[title] = texts
        del paperDict[title]['sscResult']
        
    # 分類されたアブストを出力
    with open(outputDirPath + "labeledAbst.json", 'w') as f:
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
        
        # 埋め込み
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDirPath + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
        

            
        """
        2. SciBERT
        """
        # モデルの初期化
        modelName  = "SciBert"
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
                
        # 埋め込み
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDirPath + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
        
        """
        3. SPECTER
        """
        # モデルの初期化
        modelName  = "Specter"
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter')

        # 埋め込み
        outputEmblabel = embedingLabel(tokenizer, model, paperDict, concatLabeledAbst)
        with open(outputEmbLabelDirPath + "labeledAbst" + modelName + ".json", 'w') as f:
            json.dump(outputEmblabel, f, indent=4)
            
        message = "【完了】shape-and-emmbedding-bgobj.py"
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

        # bg, method, result
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

if __name__ == '__main__':
    main()