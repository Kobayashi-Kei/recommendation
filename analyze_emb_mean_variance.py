import json
import statistics

embPath = 'dataserver/axcell/medium-pretrain_average_pooling_entire/embedding/titleAbstSpecter.json'

with open(embPath, 'r') as f:
    titleAbstEmb = json.load(f)


for title, abst_emb in titleAbstEmb.items():
    print(statistics.stdev(abst_emb))