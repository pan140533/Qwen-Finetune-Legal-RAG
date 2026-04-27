import os
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])

# 你的两个docx文件
files = [
    r'.\中华人民共和国劳动法_20181229.docx',
    r'.\中华人民共和国民法典_20200528.docx'
]

all_text = ''
for f in files:
    all_text += read_docx(f) + '\n\n'

# 简单按句号分割，每个句子作为一个训练样本（可自行调整分割逻辑）
sentences = [s.strip() for s in all_text.replace('\n', '').split('。') if len(s.strip()) > 10]

# 保存为训练数据（每行一个样本）
with open('law_train.txt', 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')

print(f'生成 {len(sentences)} 条训练样本，保存到 law_train.txt')