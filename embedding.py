import json
from transformers import *
import lineNotifier
import traceback
import os



"""
BERT・SciBERT・SPECTERなどでタイトル・アブストラクト全体を埋め込み、ファイル出力する
"""
def main():
    """
    入出力ファイルの定義
    """
    # taskデータのパスと出力パス
    #filePath = './data/axcell/axcellRecomData-old.json'
    filePath = './data/axcell/axcellRecomData.json'
    outputFilePath = filePath.replace('.json', '-includeEmb.json')

    with open(filePath, 'r') as f:
        paperList = json.load(f)

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
        for i, paper in enumerate(paperList):
            # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
            input = tokenizer(paper['abstract'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            # model(**input).pooler_output にすることも検討
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['abstract_bert_embedding'] = output[0].tolist()
            
            title_abs = paper['title'] + tokenizer.sep_token + paper['abstract']
            input = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['title_abs_bert_embedding'] = output[0].tolist()


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
        for i, paper in enumerate(paperList):
            # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
            input = tokenizer(paper['abstract'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['abstract_scibert_embedding'] = output[0].tolist()
            
            title_abs = paper['title'] + tokenizer.sep_token + paper['abstract']
            input = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['title_abs_scibert_embedding'] = output[0].tolist()
            
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
        for i, paper in enumerate(paperList):
            input = tokenizer(paper['abstract'], padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['abstract_specter_embedding'] = output[0].tolist()
            
            title_abs = paper['title'] + tokenizer.sep_token + paper['abstract']
            input = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
            output = model(**input).last_hidden_state[:, 0, :]
            paperList[i]['title_abs_specter_embedding'] = output[0].tolist()
                
        del tokenizer
        del model

        """
        ファイル出力
        """
        with open(outputFilePath, 'w') as f:
            json.dump(paperList, f, indent=4)
        message = "【完了】emmbedding.py"
        lineNotifier.line_notify(message)
            
    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)

if __name__ == "__main__":
    main()