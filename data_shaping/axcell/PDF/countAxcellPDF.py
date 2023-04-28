"""
AxcellのPDFをダウンロードしたか確認用
収集したAxcellのPDFの種類を数える
"""
from fileinput import filename


pdfFileList = []

with open('ls_-l_axcellPDF.txt') as f:
    lines = f.readlines()
    for line in lines:
        fileName = line.split()[8]
        if not fileName in pdfFileList:
            pdfFileList.append(fileName)



with open('ls_-l_axcellPDF_2.txt') as f:
    lines = f.readlines()
    for line in lines:
        fileName = line.split()[8]
        if not fileName in pdfFileList:
            pdfFileList.append(fileName)

print(len(pdfFileList))
