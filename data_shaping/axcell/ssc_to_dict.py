import json
import re


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)
            # print(data)
            # exit()
    return data_list


def transLabel(label):
    if label == 'background_label':
        return "bg"
    elif label == 'objective_label':
        return "obj"
    elif label == 'method_label':
        return "method"
    elif label == 'result_label':
        return "res"
    elif label == 'other_label':
        return "other"


file_path = "dataserver/axcell/large/inputForSSC.jsonl"
ssc_input_data = read_jsonl(file_path)

# ssc resultの読み込み
file_path = "dataserver/axcell/large/resultSSC.txt"
ssc_result_data = {}
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        data = json.loads(line)
        if ssc_input_data[i]['sentences'][0][:5] != data[1][0][0][:5]:
            print("Error")
        # print(ssc_input_data[i]['sentences'][0][:5])
        # print(data[1][0][0][:5])
        # exit()
        tmpData = []
        for sentLabelPair in data[1]:
            sentence = sentLabelPair[0]
            label = sentLabelPair[1]
            tmpData.append([sentence, transLabel(label)])
        # print(tmpData)
        # exit()
        if "What happens if" in re.sub(' +', ' ', ssc_input_data[i]
                                       ['title'].strip()):
            print(ssc_input_data[i]
                  ['title'])
            exit()

        ssc_result_data[re.sub(' +', ' ', ssc_input_data[i]
                               ['title'].strip())] = tmpData

exit()

outputPath = "dataserver/axcell/large/result_ssc.json"
with open(outputPath, "w") as f:
    json.dump(ssc_result_data, f, indent=4)

# データのリストを表示する
for data in ssc_result_data:
    print(data)
    break
# print(jsonl_data)
