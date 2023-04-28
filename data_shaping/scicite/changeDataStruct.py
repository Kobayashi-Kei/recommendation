import json





def changeData(path):
    with open(path, 'r') as f: 
        data = json.load(f)
    
    for key in data:
        if not 'citingTitle' in data[key]:
            data[key]['citingTitle'] = None
        if not 'citingAbstract' in data[key]:
            data[key]['citingAbstract'] = None
        if not 'citedTitle' in data[key]:
            data[key]['citedTitle'] = None
        if not 'citedAbstract' in data[key]:
            data[key]['citedAbstract'] = None

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        
path = "data/scicite/train-withAbst.json"
changeData(path)
path = "data/scicite/dev-withAbst.json"
changeData(path)
path = "data/scicite/test-withAbst.json"
changeData(path)