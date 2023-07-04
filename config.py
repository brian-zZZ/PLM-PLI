# SEED = 42
# 基本参数
n_gpu = 1
gpu_start = 0
gradient_accumulation = 1 # 8 # mark
lr = 5e-4 # 1e-4 # mark
weight_decay = 1e-5  # 1e-4 # mark
decay_interval = 5
lr_decay = 0.995 # 0.995 # 1 # mark
do_train = True
do_test = True
do_save_emb = True
do_save_pretrained_emb = False
return_emb = do_save_emb | do_save_pretrained_emb

# PLI任务
sampled_frac = 1
freeze_seq_encoder = False


# 加载数据配置参数
import yaml
from argparse import ArgumentParser
parser = ArgumentParser(description='Model configuration')
parser.add_argument('--SEED', type=int, default=42)
parser.add_argument('--model_name', type=str, default='tape', choices=['esm1b', 'prottrans', 'tape'])
parser.add_argument('--task', type=str, default='PDBBind', choices=['PDBBind', 'Kinase', 'DUDE', 'GPCR'])
parser.add_argument('--random', action='store_true', help='Wether random initialize model weights')
args = parser.parse_args()
args_dict = yaml.load(open("args.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
for k, v in args_dict.items():
    setattr(args, k, v)

# model_name = 'tape'
model_name = args.model_name
model_hid_dims = {
    # 'esm1b': 1280, # 1v
    'esm1b': 768, # 1b
    'prottrans': 1024,
    'tape': 768
}
max_seq_len = 512
if model_name == 'esm1b':
    # max_seq_len -= 2 # 1v
    max_seq_len -= 1 # 1b
    batch_size = 2
    # gradient_accumulation = 8
elif model_name == 'prottrans':
    batch_size = 4
elif model_name == 'tape':
    max_seq_len -= 2
    batch_size = 16
# task = 'Kinase'
task = args.task
task_metrics = {
    'PDBBind': "R", # Personr's ρ
    'Kinase': "AUC",
    'DUDE': "AUC",
    'GPCR': "AUC"
}
if task == 'PDBBind':
    epochs = 50
elif task == 'Kinase':
    epochs = 10
elif task == 'DUDE':
    epochs = 10
elif task == 'GPCR':
    epochs = 10

args.max_seq_len = max_seq_len
SEED = args.SEED
random = args.random


# 获取变量
config_variables = dict(globals(), **locals())
config_variables = {k: v for k, v in config_variables.items() if '__' not in k}
config_variables = {k: v for k, v in config_variables.items() if type(v) in [int, float, bool, str, dict, list, tuple]}
print(config_variables)
# 生成保存路径
import os
import time
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
path_model = 'outputs/%s/%s-%s/' % (model_name, task, current_time)
os.makedirs(path_model, exist_ok=True)
# 保存配置参数文件
import json
with open(path_model+"config.json", 'w') as f:
    f.write(json.dumps(config_variables, indent=2))
