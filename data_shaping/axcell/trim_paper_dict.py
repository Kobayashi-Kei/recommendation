import json
import re

path = "dataserver/axcell/large/paperDict.json"

with open(path, "r") as f:
    data = json.load(f)

retData = {}
for title, paper in data.items():
    trimTitle = re.sub(' +', ' ', title.strip())
    paper["title"] = re.sub(' +', ' ', paper["title"].strip())
    paper["abstract"] = re.sub(' +', ' ', paper["abstract"].strip())
    retCites = []
    for citeTitle in paper["cite"]:
        retCites.append(re.sub(' +', ' ', citeTitle.strip()))
    paper["cite"] = retCites
    retData[trimTitle] = paper

with open(path.replace(".json", "-new.json"), "w") as f:
    json.dump(retData, f, indent=4)
