import json

def main():
    resultFilePath_1 = "result/run-entire.py-medium-02282000.json"
    resultFilePath_2 = "result/run-labeled.py-medium-02282007.json"
    # resultFilePath_1 = "result/run-labeled.py-medium-02282007.json"
    # resultFilePath_2 = "result/run-labeled-max.py-medium-02282008.json"
    # resultFilePath_2 = "result/run-labeled.py-medium-tf-idf-title_margin2-02281959.json"
    resultFilePath_2 = "result/run-labeled.py-medium-paper_margin2-03031055.json"
    # kList = [10,20,30]
    kList = [10]
    
    # ファイルの読み込み
    with open(resultFilePath_1, "r") as f:
        result_1 = json.load(f)
        
    with open(resultFilePath_2, "r") as f:
        result_2 = json.load(f)
        
    # methodの解釈
    methodName_1 = resultFileNameToMethodName(
        "-".join(resultFilePath_1.split("/")[1].split("-")[:-1])
    )
    methodName_2 = resultFileNameToMethodName(
        "-".join(resultFilePath_2.split("/")[1].split("-")[:-1])
    )

    
    print("比較する手法: \n1", methodName_1, "\n2", methodName_2, "\n")
    for idx, k in enumerate(kList):
        
        method_1_count = 0
        method_2_count = 0
        method_1_2_count = 0
        
        allCitationCount = 0
        for i, recom in enumerate(result_1):
            for j, citePaper in enumerate(recom["result"]):
                allCitationCount += 1
                method_1_ok = False
                method_2_ok = False
                if result_1[i]["result"][j]["title"] != result_2[i]["result"][j]["title"]:
                    print("Not Same Paper Error")
                    exit()
                if result_1[i]["result"][j]["rank"] <= k:
                    method_1_count += 1
                    method_1_ok = True
                if result_2[i]["result"][j]["rank"] <= k:
                    method_2_count += 1
                    method_2_ok = True
                if method_1_ok or method_2_ok:
                    method_1_2_count += 1
                if not method_1_ok and method_2_ok:
                    print(result_1[i]["result"][j]["rank"], result_2[i]["result"][j]["rank"])
                    print(result_2[i]["queryTitle"])
                    print(result_2[i]["queryAbstract"])
                    print(result_2[i]["result"][j])
        if idx == 0:
            print("引用文献数(正解の数)", allCitationCount, "\n")
            
        print("----- Top"+ str(k), "-----")
        print(methodName_1, ": ", method_1_count)
        print(methodName_2, ": ", method_2_count)
        print(methodName_1, "or", methodName_2, ": ", method_1_2_count)
        print()


def resultFileNameToMethodName(resultFileName): 
    if resultFileName == "run-entire.py-medium":
        methodName = "SPECTER アブスト全体(ベースライン)"
    elif resultFileName == "run-labeled.py-medium":
        methodName = "SPECTER 追加学習無し 平均"
    elif resultFileName == "run-labeled-max.py-medium":
        methodName = "SPECTER 追加学習無し 最大"
    elif resultFileName == "run-labeled.py-medium-paper_margin2":
        methodName = "SPECTER 論文手法 平均"
    elif resultFileName == "run-labeled-max.py-medium-paper_margin2":
        methodName = "SPECTER 論文手法 最大"
    elif resultFileName == "run-labeled.py-medium-tf-idf-title_margin2":
        methodName = "title-tf-idf 平均"
    elif resultFileName == "run-labeled-max.py-medium-tf-idf-title_margin2":
        methodName = "title-tf-idf 最大"
    return methodName

    
if __name__ == "__main__":
    main()