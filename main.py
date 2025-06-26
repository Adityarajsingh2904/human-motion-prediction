import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def seed_torch(seed=3450):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

seed_torch()


import pandas as pd
from configs.config import args

from run import H36MRunner, CMURunner
from datas_dct import define_actions, define_actions_cmu

print("\n================== Arguments =================")
print(vars(args))
print("==========================================\n")

if args.dct == True:
    if args.exp_name == "h36m":
        r = H36MRunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=args.dct_n,
                       device=args.device, num_works=args.num_works,
                       test_manner=args.test_manner, debug_step=args.debug_step,
                       batch_size=args.batch_size, epochs=args.epochs,
                       learning_rate=args.learning_rate)
        acts = define_actions("all")

    elif args.exp_name == "cmu":
        r = CMURunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=args.dct_n,
                      device=args.device, num_works=args.num_works,
                      test_manner=args.test_manner, debug_step=args.debug_step,
                      batch_size=args.batch_size, epochs=args.epochs,
                      learning_rate=args.learning_rate)
        acts = define_actions_cmu("all")

else:
    if args.exp_name == "h36m":
        r = H36MRunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=0,
                       device=args.device, num_works=args.num_works,
                       test_manner=args.test_manner, debug_step=args.debug_step,
                       batch_size=args.batch_size, epochs=args.epochs,
                       learning_rate=args.learning_rate)
        acts = define_actions("all")

    elif args.exp_name == "cmu":
        r = CMURunner(exp_name=args.exp_name, input_n=args.input_n, output_n=args.output_n, dct_n=0,
                      device=args.device, num_works=args.num_works,
                      test_manner=args.test_manner, debug_step=args.debug_step,
                      batch_size=args.batch_size, epochs=args.epochs,
                      learning_rate=args.learning_rate)
        acts = define_actions_cmu("all")

if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
else:
    errs = r.test()

    col = r.cfg.frame_ids
    d = pd.DataFrame(errs, index=acts, columns=col)
    d.to_csv(f"{r.cfg.exp_name}_in{r.cfg.input_n}out{r.cfg.output_n}dctn{r.cfg.dct_n}_{r.cfg.test_manner}.csv", line_terminator="\n")




