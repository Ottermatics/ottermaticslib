import os
import pdfplumber
import PyPDF2

'''We should combine this with `Location` to categorize research across projects'''

pdfdir = os.path.join(os.curdir,'Triton','research')
pdffile = os.path.join(pdfdir,'1-s2.0-S0921344919305750-main.pdf')


def get_pdf_contents(pdffile):
    content = {'pages':{},'meta':None}
    with pdfplumber.open(pdffile) as pdf:
        content['meta'] = pdf.metadata
        for page in pdf.pages:
            content['pages'][page.page_number] = page.extract_text()
        #TODO: Objects / Tables Ect
    return content

pdf_data = {}

for root,dirs,fils in os.walk(pdfdir):
    for fil in fils:
        try:
            filepath= os.path.join(root,fil)
            print(f'getting {filepath}')
            pdf_data[filepath] = get_pdf_contents( filepath)
        except Exception as e:
            print(e)
            
