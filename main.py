from utils.functions import *
from trainer import Trainer


def main(args):
    trainer = Trainer(args)
    if not args.eval_only:
        trainer.train()
    else:
        trainer.evaluate_wo_train()
    # else:
        # trainer.train_fw()


if __name__ == "__main__":
    args = build_args()
    # if args.dataset_name in ["arxiv", "products"]:
    #     if args.few_shot_setting == "3,3":
    #         args.few_shot_setting = "5,3"
    #     elif args.few_shot_setting == "3,5":
    #         args.few_shot_setting = "5,5"
    #     elif args.few_shot_setting == "3,10":
    #         args.few_shot_setting = "10,3"
    if args.dataset_name not in args.eval_model_path:
        args.eval_model_path = f"./output/{args.dataset_name}/{args.eval_model_path}"
    N = int(args.few_shot_setting.split(",")[0])
    if args.label_as_init and args.token_num < N:
        args.token_num = N
    main(args)
