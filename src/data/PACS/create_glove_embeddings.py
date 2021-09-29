import numpy as np
import os
import pickle

root_path = '/media/mvp18/Study/ZS-SBIR/datasets/'
PACS_img_path = root_path+'PACS/Raw images/kfold/'

classes = sorted(os.listdir(os.path.join(PACS_img_path, os.listdir(PACS_img_path)[0])))

glove_dict_path = root_path+'glove.6B/glove.6B.300d.txt'
embeddings_dict = {}
with open(glove_dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

clss_embed = {}
for clss in classes:
	
	clss_embed[clss] = embeddings_dict[clss]

print(clss_embed.keys())

with open('PACS_glove300.pkl', 'wb') as f:
	pickle.dump(clss_embed, f)