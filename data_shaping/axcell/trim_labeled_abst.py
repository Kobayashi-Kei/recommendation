import json
import re

path = "dataserver/axcell/large/labeledAbst.json"

with open(path, "r") as f:
    data = json.load(f)

retData = {}
for title, paper in data.items():
    trimTitle = re.sub(' +', ' ', title.strip())
    retData[trimTitle] = paper

with open(path.replace(".json", "-new.json"), "w") as f:
    json.dump(retData, f, indent=4)
