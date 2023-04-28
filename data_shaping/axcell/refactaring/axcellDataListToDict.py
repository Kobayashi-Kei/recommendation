import json


path = "./data/axcell/axcellRecomData-includeEmb-includeLabelEmb.json"
with open(path, 'r') as f:
    data = json.load(f)
    
outputDict = {}
for item in data:
    outputDict[item['title']] = item

with open(path.replace('.json', '-dict.json'), 'w') as f:
    json.dump(outputDict, f, indent=4)