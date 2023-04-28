import json


datasetName = "dev"
dataPath = "dataserver/scicite/"+ datasetName + "/main.json"
with open(dataPath, 'r') as f:
    data = json.load(f)
    
outputPaperDict = {}
for key in data:
    if data[key]['citingTitle'] != None and data[key]['citingAbstract'] != None \
        and data[key]['citedTitle'] != None and data[key]['citedAbstract'] != None:
        outputPaperDict[data[key]["citingPaperId"]] = {
            "title": data[key]["citingTitle"],
            "abstract": data[key]["citingAbstract"]
        }
        outputPaperDict[data[key]["citedPaperId"]] = {
            "title": data[key]["citedTitle"],
            "abstract": data[key]["citedAbstract"]
        }

outputPath = "dataserver/scicite/"+ datasetName + "/paperDict.json"
with open(outputPath, 'w') as f:
    json.dump(outputPaperDict, f, indent=4)