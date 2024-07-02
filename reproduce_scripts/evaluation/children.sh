LM_PATH='./LMs'

# dataset_name: children, few_shot_setting: 3,3
python main.py --token_num 3 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name children --label_as_init True --few_shot_setting 3,3 --cut_off 128 --emb_type LM --lm_path $LM_PATH 

# dataset_name: children, few_shot_setting: 3,5
python main.py --token_num 3 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name children --label_as_init True --few_shot_setting 3,5 --cut_off 128 --emb_type LM --lm_path $LM_PATH 

# # dataset_name: children, few_shot_setting: 3,10
python main.py --token_num 3 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name children --label_as_init True --few_shot_setting 3,10 --cut_off 128 --emb_type LM --lm_path $LM_PATH
