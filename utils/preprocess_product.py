import gc
import os
import time
from tqdm import tqdm
import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
from types import SimpleNamespace as SN


from utils.functions import sample_nodes
from settings import *


def _load_ogb_products(args, labels):
    from ogb.utils.url import download_url, extract_zip
    

    import gdown
    
    
    
    
    
    
    
    
    
    opath = DATA_INFO[args.dataset_name]["data_root"]
    
    output = os.path.join(opath, 'Amazon-3M.raw.zip')
    if not os.path.exists(output):
        url = DATA_INFO[args.dataset_name]["raw_text_url"]
        gdown.download(url=url, output=output, quiet=False, fuzzy=False)
    if not os.path.exists(os.path.join(opath, "Amazon-3M.raw")):
        extract_zip(output, opath)
    raw_text_path = os.path.join(opath, "Amazon-3M.raw")

    def read_mappings(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2productcategory.csv.gz"
        products_asin_path_csv = f"{data_root}/mapping/nodeidx2asin.csv.gz"  
        products_ids = pd.read_csv(products_asin_path_csv)
        categories = pd.read_csv(category_path_csv)
        
        return categories, products_ids  

    def get_mapping_product(labels, meta_data: pd.DataFrame, products_ids: pd.DataFrame, categories):
        
        products_ids.columns = ["ID", "asin"]
        categories.columns = ["label_idx", "category"]  
        meta_data.columns = ['asin', 'title', 'content']
        products_ids["label_idx"] = labels
        data = pd.merge(products_ids, meta_data, how="left", on="asin")  
        data = pd.merge(data, categories, how="left", on="label_idx")  
        
        return data

    def read_product_json(raw_text_path):
        import json
        import gzip
        if not os.path.exists(os.path.join(raw_text_path, "trn.json")):
            trn_json = os.path.join(raw_text_path, "trn.json.gz")
            trn_json = gzip.GzipFile(trn_json)
            open(os.path.join(raw_text_path, "trn.json"), "wb+").write(trn_json.read())
            os.unlink(os.path.join(raw_text_path, "trn.json.gz"))
            tst_json = os.path.join(raw_text_path, "tst.json.gz")
            tst_json = gzip.GzipFile(tst_json)
            open(os.path.join(raw_text_path, "tst.json"), "wb+").write(tst_json.read())
            os.unlink(os.path.join(raw_text_path, "tst.json.gz"))
            os.unlink(os.path.join(raw_text_path, "Yf.txt"))  

        i = 1
        for root, dirs, files in os.walk(os.path.join(raw_text_path, '')):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf_8_sig') as file_in:
                    title = []
                    for line in file_in.readlines():
                        dic = json.loads(line)

                        dic['title'] = dic['title'].strip("\"\n")
                        title.append(dic)
                    name_attribute = ["uid", "title", "content"]
                    writercsv = pd.DataFrame(columns=name_attribute, data=title)
                    writercsv.to_csv(os.path.join(raw_text_path, f'product' + str(i) + '.csv'), index=False,
                                     encoding='utf_8_sig')  
                    i = i + 1
        return

    def read_meta_product(raw_text_path):
        
        if not os.path.exists(os.path.join(raw_text_path, f'product3.csv')):
            read_product_json(raw_text_path)  
            path_product1 = os.path.join(raw_text_path, f'product1.csv')
            path_product2 = os.path.join(raw_text_path, f'product2.csv')
            pro1 = pd.read_csv(path_product1)
            pro2 = pd.read_csv(path_product2)
            file = pd.concat([pro1, pro2])
            file.drop_duplicates()
            file.to_csv(os.path.join(raw_text_path, f'product3.csv'), index=False, sep=" ")
        else:
            file = pd.read_csv(os.path.join(raw_text_path, 'product3.csv'), sep=" ")

        return file

    print('Loading raw text')
    meta_data = read_meta_product(raw_text_path)  
    categories, products_ids = read_mappings(DATA_INFO[args.dataset_name]["data_root"])
    node_data = get_mapping_product(labels, meta_data, products_ids, categories)  
    import gc
    del meta_data, categories, products_ids
    text_func = {
        'T': lambda x: x['title'],
        'TC': lambda x: f"Title: {x['title']}. Content: {x['content']}",
    }
    node_data['text'] = node_data.apply(text_func[args.process_mode], axis=1)
    node_data['text'] = node_data.apply(lambda x: ' '.join(str(x['text']).split(' ')[:args.cut_off]), axis=1)
    node_data = node_data[['ID', 'text']]  
    return node_data


def _tokenize_ogb_product(args, labels):
    tokenizer = AutoTokenizer.from_pretrained(args.lm_type, cache_dir=args.lm_path)
    node_text = _load_ogb_products(args, labels)
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    
    node_chunk_size = 1000000
    max_length = 512
    data_info =  DATA_INFO[args.dataset_name]
    info = {
        'input_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=np.uint16),
        'attention_mask': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool),
        'token_type_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool)
    }
    token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
    token_folder = f"{DATA_PATH}{args.dataset_name}/{args.lm_type.split('/')[-1]}/"
    for k, k_info in info.items():
        k_info.path = f'{token_folder}{k}.npy'
    shape = (data_info['n_nodes'], max_length)
    
    token_keys = tokenizer(get_text([0]), padding='max_length', truncation=True, max_length=512).data.keys()
    
    x = {k: np.memmap(info[k].path, dtype=info[k].type, mode='w+', shape=info[k].shape)
         for k in token_keys}

    for i in tqdm(range(0, shape[0], node_chunk_size)):
        j = min(i + node_chunk_size, shape[0])
        tokenized = tokenizer(get_text(range(i, j)), padding='max_length', truncation=True, max_length=512).data
        for k in token_keys:
            x[k][i:j] = np.array(tokenized[k], dtype=info[k].type)
    
