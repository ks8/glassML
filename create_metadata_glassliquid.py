""" Create data folder with metadata """
import json
import os
import re
data = []
for root, subfolders, files in os.walk('data'):
	for f in files:
		if f[-4:] == '.png' in f:
			path = os.path.join(root, f)
			category = re.findall(r'glass', path)
			if len(category) == 1:
				material = 'glass'
			else:
				material = 'liquid'
			data.append({'path': path, 'label': material})
if not os.path.exists('metadata'):
	os.makedirs('metadata')
json.dump(data, open('metadata/metadata.json', 'w'), indent=4, sort_keys=True)