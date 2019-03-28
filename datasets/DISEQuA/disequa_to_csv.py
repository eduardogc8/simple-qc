import pandas as pd
import xml.etree.ElementTree as ET


patch = "DISEQuA_v1.0.xml"


tree = ET.parse(patch)
root = tree.getroot()

datas = []
for qa in root:
	class_ = qa.attrib['type']
	for language in qa:
		ling = language.attrib['val']
		for question in language:
			if question.tag == 'question':
				text = question.text.replace('\n', '').replace('\t', '').strip()
				data = {'question': text, 'language': ling, 'class': class_}
				datas.append(data)
df = pd.DataFrame(datas)
df.to_csv('disequa.csv')