import json


outputDir = "data/newAxcell/medium/"
"""
論文のメタデータとアブストを分離してファイル出力
"""
# path = "data/axcell/small/axcellRecomData-old-includeLabeled.json"
# with open(path , 'r') as f:
#     data = json.load(f)

# paperData = {}
# labeledAbstData = {}
# for item in data:
#     labeledAbst = {}
#     labeledAbst['bg'] = item["labeledAbstract"]['backgroundText']
#     labeledAbst['obj'] = item["labeledAbstract"]['objectiveText']
#     labeledAbst['method'] = item["labeledAbstract"]["methodText"]
#     labeledAbst["res"] = item["labeledAbstract"]["resultText"]
#     labeledAbst["other"] = item["labeledAbstract"]["otherText"]
#     del item["labeledAbstract"]
#     paperData[item["title"]] = item
#     labeledAbstData[item["title"]] = labeledAbst
    
# outputPath = outputDir + "paperDict.json"
# with open(outputPath, 'w') as f:
#     json.dump(paperData, f, indent=4)

# outputPath = outputDir + "labeledAbst.json"
# with open(outputPath, 'w') as f:
#     json.dump(labeledAbstData, f, indent=4)


"""
タイトルとアブストラクトの埋め込みを整理して個別でファイルに出力
"""    
# path = "data/axcell/small/axcellRecomData-old-includeEmb.json"
# with open(path , 'r') as f:
#     data = json.load(f)

# abstBert = {}
# titleAbstBert = {}
# abstSciBert = {}
# titleAbstSciBert = {}
# abstSpecter = {}
# titleAbstSpecter = {}

# for item in data:
#     abstBert[item["title"]] = item["abstract_bert_embedding"]
#     titleAbstBert[item["title"]] = item["title_abs_bert_embedding"]
#     abstSciBert[item["title"]] = item["abstract_scibert_embedding"]
#     titleAbstSciBert[item["title"]] = item["title_abs_scibert_embedding"]
#     abstSpecter[item["title"]] = item["abstract_specter_embedding"]
#     titleAbstSpecter[item["title"]] = item["title_abs_specter_embedding"]


outputEmbDir = outputDir + "embedding/" 
# name = "abstBert"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(abstBert, f, indent=4)

# name = "titleAbstBert"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(titleAbstBert, f, indent=4)

# name = "abstSciBert"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(abstSciBert, f, indent=4)

# name = "titleAbstSciBert"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(titleAbstSciBert, f, indent=4)

# name = "abstSpecter"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(abstSpecter, f, indent=4)

# name = "titleAbstSpecter"
# path = outputEmbDir + name + ".json"
# with open(path, 'w') as f:
#     json.dump(titleAbstSpecter, f, indent=4)


"""
タイトルと分類されたアブストラクトの埋め込みを整理して個別でファイルに出力
"""    
path = "data/axcell/small/axcellRecomData-old-includeEmb-includeLabelEmb.json"
with open(path , 'r') as f:
    data = json.load(f)

labeledAbstBert = {}
labeledAbstSciBert = {}
labeledAbstSpecter = {}

for item in data:
    labeledAbstBert[item["title"]] = {}
    labeledAbstBert[item["title"]]["title"] = item["labeled_abstract_bert_embedding"]["title"]
    labeledAbstBert[item["title"]]["bg"] = item["labeled_abstract_bert_embedding"]["backgroundText"]
    labeledAbstBert[item["title"]]["obj"] = item["labeled_abstract_bert_embedding"]["objectiveText"]
    labeledAbstBert[item["title"]]["method"] = item["labeled_abstract_bert_embedding"]["methodText"]
    labeledAbstBert[item["title"]]["res"] = item["labeled_abstract_bert_embedding"]["resultText"]
    labeledAbstBert[item["title"]]["other"] = item["labeled_abstract_bert_embedding"]["otherText"]
    
    labeledAbstSciBert[item["title"]] = {}
    labeledAbstSciBert[item["title"]]["title"] = item["labeled_abstract_scibert_embedding"]["title"]
    labeledAbstSciBert[item["title"]]["bg"] = item["labeled_abstract_scibert_embedding"]["backgroundText"]
    labeledAbstSciBert[item["title"]]["obj"] = item["labeled_abstract_scibert_embedding"]["objectiveText"]
    labeledAbstSciBert[item["title"]]["method"] = item["labeled_abstract_scibert_embedding"]["methodText"]
    labeledAbstSciBert[item["title"]]["res"] = item["labeled_abstract_scibert_embedding"]["resultText"]
    labeledAbstSciBert[item["title"]]["other"] = item["labeled_abstract_scibert_embedding"]["otherText"]
    
    labeledAbstSpecter[item["title"]] = {}
    labeledAbstSpecter[item["title"]]["title"] = item["labeled_abstract_specter_embedding"]["title"]
    labeledAbstSpecter[item["title"]]["bg"] = item["labeled_abstract_specter_embedding"]["backgroundText"]
    labeledAbstSpecter[item["title"]]["obj"] = item["labeled_abstract_specter_embedding"]["objectiveText"]
    labeledAbstSpecter[item["title"]]["method"] = item["labeled_abstract_specter_embedding"]["methodText"]
    labeledAbstSpecter[item["title"]]["res"] = item["labeled_abstract_specter_embedding"]["resultText"]
    labeledAbstSpecter[item["title"]]["other"] = item["labeled_abstract_specter_embedding"]["otherText"]
    

outputEmbLabeledDir = outputDir + "embLabel/"    

name = "labeledAbstBert"
path = outputEmbLabeledDir + name + ".json"
with open(path, 'w') as f:
    json.dump(labeledAbstBert, f, indent=4)

name = "labeledAbstSciBert"
path = outputEmbLabeledDir + name + ".json"
with open(path, 'w') as f:
    json.dump(labeledAbstSciBert, f, indent=4)

name = "labeledAbstSpecter"
path = outputEmbLabeledDir + name + ".json"
with open(path, 'w') as f:
    json.dump(labeledAbstSpecter, f, indent=4)

