import json
import traceback
import lineNotifier
from transformers import *
import os
from Specter import Specter
import time

"""
アブストラクトの分類結果を独自データで追加学習したSPECTERを用いて埋め込みをする
"""	
def main():
    """
    データセットのパスなどを代入
    """
    # 用いる観点をリストで入力
    labelList = ["title","bg", "obj", "method", "res"]
    # 用いるモデルパラメータのパスを入力（labelListと同じ順序で）
    modelCheckpointsPathList = [
        "docker_specter/source/pytorch_lightning_training_script/save/version_1/checkpoints/title-ep-epoch=3_avg_val_loss-avg_val_loss=0.127.ckpt",
        "docker_specter/source/pytorch_lightning_training_script/save/version_1/checkpoints/bg-ep-epoch=3_avg_val_loss-avg_val_loss=0.111.ckpt",
        "docker_specter/source/pytorch_lightning_training_script/save/version_1/checkpoints/obj-ep-epoch=3_avg_val_loss-avg_val_loss=0.285.ckpt",
        "docker_specter/source/pytorch_lightning_training_script/save/version_1/checkpoints/method-ep-epoch=3_avg_val_loss-avg_val_loss=0.082.ckpt",
        "docker_specter/source/pytorch_lightning_training_script/save/version_1/checkpoints/res-ep-epoch=3_avg_val_loss-avg_val_loss=0.146.ckpt"
    ]
        
    size = "medium"
    method = "specterPaper"
    titleOrAbst = "abst"
   
    # 入力
    dirPath = "../dataserver/axcell/" + size
    dataPath = dirPath + "/paperDict.json"
    labeledAbstPath = dirPath + "/labeledAbst.json"
    
    # 出力
    outputDirPath = dirPath + "-" + method + "/"
    # outputDirPath = dirPath + titleOrAbst + "-" + method + "/" 
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
    
    # 分類されたアブストラクトをロード
    with open(labeledAbstPath, 'r') as f:
        labeledAbstDict = json.load(f)
    # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
    for title in labeledAbstDict:
        labeledAbstDict[title]["title"] = title

    try:
        """
        観点毎のデータで学習した観点毎のSPECTERモデルで埋め込み
        """
        # 出力用
        labeledAbstSpecter = {}
        
        for i, label in labelList:
            # モデルの初期化
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            model = Specter.load_from_checkpoint(modelCheckpointsPathList[i])
            # model = AutoModel.from_pretrained('allenai/specter')

            model.cuda()
            model.eval()

            
            count = 0
            # 埋め込み
            for title, paper in paperDict.items():
                if not title in labeledAbstSpecter:
                    labeledAbstSpecter[title] = {}
                
                if labeledAbstDict[title][label]:
                    input = tokenizer(
                        labeledAbstDict[title][label], # 文の間には[SEP]を挿入しない（挿入した方が良かったりする？）
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length=512
                        ).to('cuda')                 

                    count += 1
                    
                    # print(input)
                    output = model(**input)[0].tolist()
                    # print(output)
                    print(count, labeledAbstDict[title][label])
                    labeledAbstSpecter[title][label] = output

                else:
                    labeledAbstSpecter[title][label] = None
                   
        # ファイル出力
        with open(outputEmbLabelDirPath + "labeledAbstSpecter.json", 'w') as f:
            json.dump(labeledAbstSpecter, f, indent=4)
            
        message = "【完了】shape-and-emmbedding.py"
        lineNotifier.line_notify(message)
            
    except Exception as e:
        print(traceback.format_exc())
        message = "Error: " + os.path.basename(__file__) + " " + str(traceback.format_exc())
        lineNotifier.line_notify(message)


if __name__ == '__main__':
    main()