import json


"""
観点毎のデータの偏りが無いかを確かめるために
データセットに含まれる総単語数を観点毎に数え上げる
"""
def main(): 
    # size = "train"
    # path = "dataserver/scicite/" + size + "/labeledAbst.json"
    size = "large"
    path = "dataserver/axcell/" + size + "/labeledAbst.json"
    with open(path, 'r') as f:
        labeledAbstDict = json.load(f)

    labelList = ['bg', 'obj', 'method', 'res', 'other']
    labelWordCount = {label: 0 for label in labelList}

    for paper in labeledAbstDict.values():
        for key, value in paper.items():
            if len(value) > 0:
                labelWordCount[key] += len(value.split(" "))
        #         print(key)
        #         print(value)
        #         print(len(value.split(" ")))
        #         print(labelWordCount)
        # exit()
    print(labelWordCount)

if __name__ == "__main__":
    main()