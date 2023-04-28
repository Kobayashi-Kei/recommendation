import json
import nltk
nltk.download('punkt')

"""
以下の形式にしてファイル出力する
{
    "abstract_id": 0, 
    "sentences":[
        ~,
        ~
    ]
}
"""
def main():
    filePath = 'dataserver/scicite/train/main.json'
    shapeData(filePath)
    filePath = 'dataserver/scicite/dev/main.json'
    shapeData(filePath)
    filePath = 'dataserver/scicite/test/main.json'
    shapeData(filePath)

def shapeData(path):
    fileName = path.split('/')[-1]
    dirPath = path.replace(fileName, '')
    outputFileName = fileName.replace('main.json', 'sscInput.jsonl')


    with open(path, 'r') as f:
        paperList = json.load(f)

    count = 0
    sscLineNumToPaperId = {}

    for key in paperList:    
        if paperList[key]['citingTitle'] != None and paperList[key]['citingAbstract'] != None \
            and paperList[key]['citedTitle'] != None and paperList[key]['citedAbstract'] != None:
                
            # ターゲット論文
            outputDict = {"abstract_id": 0}
            outputDict['title'] = paperList[key]['citingTitle']
            sentences = nltk.sent_tokenize(paperList[key]['citingAbstract'])
            outputDict['sentences'] = sentences
            # SSCへの入力用ファイルにアブストラクトを出力
            with open(dirPath+outputFileName, 'a') as f:
                json.dump(outputDict, f)
                print('', file=f)
            # SSCの何行目の分類結果がどの論文かを表す辞書
            sscLineNumToPaperId[count] = paperList[key]['citingPaperId']
            count += 1
        
            # 被引用論文
            outputDict = {"abstract_id": 0}
            outputDict['title'] = paperList[key]['citedTitle']
            sentences = nltk.sent_tokenize(paperList[key]['citedAbstract'])
            outputDict['sentences'] = sentences
            # SSCへ入力用ファイルにアブストラクトを出力
            with open(dirPath+outputFileName, 'a') as f:
                json.dump(outputDict, f)
                print('', file=f)
            # SSCの何行目の分類結果がどの論文かを表す辞書
            sscLineNumToPaperId[count] = paperList[key]['citedPaperId']
            count += 1
            
    # print(sscLineNumToPaperId)
    with open(dirPath + outputFileName.replace('sscInput.jsonl', 'sscInputLineNumToPaperId.json'), 'w') as f:
        json.dump(sscLineNumToPaperId, f, indent=4)

if __name__ == "__main__":
    main()
