import json



with open("dataserver/axcell/medium/paperDict.json" , "r") as f:
    mediumPaperDict = json.load(f)
    

with open("dataserver/axcell/large/specter/paper/taskData.json") as f:
    taskData = json.load(f)
    

for title, citeDict in taskData.items():
    
    if title in mediumPaperDict:
        print("found")
    
    for citeTitle, citeCiteTitleList in citeDict.items():
        if citeTitle in mediumPaperDict:
            print("found")
        for citeCiteTitle in citeCiteTitleList:
            if citeCiteTitle in mediumPaperDict:
                print("found")
    
with open("dataserver/axcell/large/specter/Specter-abst/taskData.json") as f:
    taskData = json.load(f)
    

for title, citeDict in taskData.items():
    # print(citeDict)
    
    if title in mediumPaperDict:
        print("found")
    
    for citeTitle, citeCiteTitleList in citeDict.items():
        # print(citeCiteTitleList)
        if citeTitle in mediumPaperDict:
            print("found")
        for citeCiteTitle in citeCiteTitleList:
            if citeCiteTitle in mediumPaperDict:
                print("found")
    # exit()
# if "Fully connected deep structured networks" in mediumPaperDict:
#     print("test found")