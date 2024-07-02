
OGB_ROOT = './dataset/'
AMAZON_ROOT = './dataset/amazon/'
DATA_PATH = 'data/'

DATA_INFO = {
    'arxiv': {
        'type': 'ogb',
        'train_ratio': 0,  
        'n_labels': 40,
        'n_nodes': 169343,
        'data_name': 'ogbn-arxiv',
        'raw_data_path': OGB_ROOT,  
        'max_length': 512,  
        'data_root': f'{OGB_ROOT}ogbn_arxiv',  
        'raw_text_url': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
    },
    'products': (product_settings := {
        'type': 'ogb',
        'train_ratio': 0,  
        'n_labels': 47,
        'n_nodes': 2449029,
        'max_length': 512,
        'data_name': 'ogbn-products',  
        'download_name': 'AmazonTitles-3M',
        'raw_data_path': OGB_ROOT,  
        'data_root': f'{OGB_ROOT}ogbn_products/',  
        'raw_text_url': 'https://drive.google.com/u/0/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN&export=download'
    }),
    'products256': {
        **product_settings,
        'cut_off': 256
    },
    
    'computers': {
        'type': 'amazon',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'train_year': 2017,
        'val_year': 2018,
        'splits': 'time',
        'n_labels': 10,
        'n_nodes': 87229,
        'data_name': 'Electronics-Computers',
        'max_length': 256,  
        'data_root': f'{AMAZON_ROOT}Electronics/Computers/',  
    },
    'children': {
        'type': 'amazon',
        'train_ratio': 0.6,
        'val_ratio': 0.2,  
        'splits': 'random',
        'n_labels': 24,
        'n_nodes': 76875,
        'data_name': 'Books-Children',
        'max_length': 256,  
        'data_root': f'{AMAZON_ROOT}Books/Children/',  
    },
    'history': {
        'type': 'amazon',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'splits': 'random',
        'n_labels': 13,
        'n_nodes': 41551,
        'data_name': 'Books-History',
        'max_length': 256,  
        'data_root': f'{AMAZON_ROOT}Books/History/',  
    },
    'photo': {
        'type': 'amazon',
        'train_ratio': 0,  
        'n_labels': 12,
        'n_nodes': 48362,
        'data_name': 'Electronics-Photo',
        'max_length': 512,  
        'splits': 'time',
        'train_year': 2015,
        'val_year': 2016,
        'data_root': f'{AMAZON_ROOT}Electronics/Photo/',
    },
}