import json
import re

prePath = "dataserver/axcell/large/specter/paper/triple"
suffixes = ["train", "dev", "test"]

for suffix in suffixes:
    path = prePath + "-" + suffix + ".json"

    with open(path, "r") as f:
        data = json.load(f)

    retData = []
    for triple in data:
        for key in triple:
            trimTitle = re.sub(' +', ' ', triple[key].strip())
            triple[key] = trimTitle
        retData.append(triple)

    with open(path.replace(".json", "-new.json"), "w") as f:
        json.dump(retData, f, indent=4)
