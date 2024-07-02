LM_PATH='./LMs'

# dataset_name: arxiv, few_shot_setting: 5,3
python main.py --token_num 5 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name arxiv --label_as_init True --few_shot_setting 5,3 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt

# dataset_name: arxiv, few_shot_setting: 5,5
python main.py --token_num 5 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name arxiv --label_as_init True --few_shot_setting 5,5 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt

# dataset_name: arxiv, few_shot_setting: 10,3
python main.py --token_num 10 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name arxiv --label_as_init True --few_shot_setting 10,3 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt --prompt_epochs 10

# dataset_name: arxiv, few_shot_setting: 10,5
python main.py --token_num 10 --LM_as_init True --num_repeat 5 --prompt_type only_first --dataset_name arxiv --label_as_init True --few_shot_setting 10,5 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt

