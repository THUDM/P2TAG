LM_PATH='./LMs'

# dataset_name: photo, few_shot_setting: 3,3
python main.py --token_num 3 --LM_as_init False --num_repeat 5 --prompt_type only_first --dataset_name photo --label_as_init True --few_shot_setting 3,3 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt

# dataset_name: photo, few_shot_setting: 3,5
python main.py --token_num 3 --LM_as_init False --num_repeat 5 --prompt_type only_first --dataset_name photo --label_as_init True --few_shot_setting 3,5 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt

# dataset_name: photo, few_shot_setting: 3,10
python main.py --token_num 3 --LM_as_init False --num_repeat 5 --prompt_type only_first --dataset_name photo --label_as_init True --few_shot_setting 3,10 --cut_off 128 --emb_type LM --lm_path $LM_PATH --eval_model_path model_deberta-base_2.pt
