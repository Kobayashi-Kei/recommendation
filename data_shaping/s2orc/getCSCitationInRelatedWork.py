"""
getCSHasPDFMetaData.pyで収集したメタデータのPDFの解析テキストを利用して
関連研究の章の被引用文献を抽出する
"""
import json
import glob
import json
import re
import lineNotifier
from tqdm import tqdm

def getCitaionInRelatedWork(paper):
    refIdList = []
    for section in paper['body_text']:
        if re.match('[I0-9\s\.]*related works?', section['section'], re.IGNORECASE):
            for cite_span in section['cite_spans']:
                if not cite_span['ref_id'] in refIdList:
                    refIdList.append(cite_span['ref_id'])
    citationTitleList = []
    for refId in refIdList:
        if refId in paper['bib_entries']:
            citationTitleList.append(paper['bib_entries'][refId]['title'])
    
    return citationTitleList
        
# feel free to wrap this into a larger loop for batches 0~99

for BATCH_ID in range(0,100):
    # create a lookup for the pdf parse based on paper ID
    paper_id_to_pdf_parse = {}
    with open(f'/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/pdf_parses/pdf_parses_{BATCH_ID}.jsonl') as f_pdf:
        for line in f_pdf:
            pdf_parse_dict = json.loads(line)
            paper_id_to_pdf_parse[pdf_parse_dict['paper_id']] = pdf_parse_dict

    # filter papers using metadata values
    csPaperList = [] 
    with open(f'/home/kobayashi/dataserver/2021-B4/kobayashi/20200705v1/full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            #print(f"Currently viewing S2ORC paper: {paper_id}")
            
            # CS分野であること
            #print(metadata_dict['mag_field_of_study'])
            if metadata_dict['mag_field_of_study'] == None or not ['Computer Science'] == metadata_dict['mag_field_of_study']:
                continue
            
            #print(paper_id)
            # PDFがあり、本文があり、参考文献があること
            if metadata_dict['has_pdf_parse'] == False or metadata_dict['has_pdf_parsed_body_text'] == False or metadata_dict['has_pdf_parsed_bib_entries'] == False:
                continue
            
            # アブストラクトがあること
            if metadata_dict['abstract'] == None and metadata_dict['has_pdf_parsed_abstract'] == False:
                continue

            # get citation context (paragraphs)!
            if paper_id in paper_id_to_pdf_parse:
                pdf_parse = paper_id_to_pdf_parse[paper_id]
                
                metadata_dict['citingPaperTitleListInRelatedWork'] = getCitaionInRelatedWork(pdf_parse)
                if len(metadata_dict['citingPaperTitleListInRelatedWork']) > 0:
                    csPaperList.append(metadata_dict)               
               

    if BATCH_ID % 20 == 0:
        message = "S2ORCのCS分野関連研究の章で引用している文献タイトルリスト収集(axcellExp/data_shaping/s2orc/getCSCitationInRelatedWork.py)がBATCH_ID = " + str(BATCH_ID)
        lineNotifier.line_notify(message)

    outputFilePath = '/home/kobayashi/paper-recom/axcellExp/data/s2orc/batch_cs_pdf_abst_bodytext_with_relatedwork_citation/s2orc-cs-with-related-work-citaion_' + str(BATCH_ID) + '.json'

    with open(outputFilePath, 'w') as f:
        json.dump(csPaperList, f, indent=4)