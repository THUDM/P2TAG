LM_PATH='./LMs'

# dataset_name: products, few_shot_setting: 5,5
python main.py --token_num 5 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name products --label_as_init True --few_shot_setting 5,3 --cut_off 128 --emb_type LM --lm_path $LM_PATH --prompt_epochs 20

# # dataset_name: products, few_shot_setting: 5,10
python main.py --token_num 5 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name products --label_as_init True --few_shot_setting 5,5 --cut_off 128 --emb_type LM --lm_path $LM_PATH --prompt_epochs 20

# # dataset_name: products, few_shot_setting: 10,3
python main.py --token_num 10 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name products --label_as_init True --few_shot_setting 10,3 --cut_off 128 --emb_type GNN --lm_path $LM_PATH 

# # dataset_name: products, few_shot_setting: 10,5
python main.py --token_num 10 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name products --label_as_init True --few_shot_setting 10,5 --cut_off 128 --emb_type GNN --lm_path $LM_PATH 

