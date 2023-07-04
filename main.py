import torch
import numpy as np
import random as rand
import pandas as pd
import os
import time
import esm
import copy
import torch.nn as nn
from train_test import *
from utils import plot_train_dev_metric
from data_handler import prepare_data

from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from transformers import T5Tokenizer
from tape import TAPETokenizer
from config import *
from models import *


if __name__ == "__main__":
    rand.seed(SEED)
    torch.manual_seed(SEED)

    dataset_pack, num_atom_features, num_bond_features = prepare_data(args, task, sampled_frac)
    if task == 'PDBBind':
        train_set, valid_set, test_set, test_casf2013_set, test_astex_set = dataset_pack
    elif task in ['Kinase', 'DUDE', 'GPCR']:
        # Kianse和DUDE只有train和test
        train_set, test_set = dataset_pack
        valid_set = test_set

    if torch.cuda.is_available():
        device_ids = []
        device = torch.device("cuda:{}".format(gpu_start))
        for i in range(n_gpu):
            device_ids.append(gpu_start+i)
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    assert batch_size >= n_gpu, "Batch size must be greater than the number of GPUs used!!!"

    """ create model, trainer and tester """
    if model_name == 'esm1b':
        # encoder, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # tokenizer = alphabet.get_batch_converter()
        # num_layers = 33
        # pro_hid_dim = model_hid_dims[model_name]
        # model = ESM1bPLI(encoder, num_layers, pro_hid_dim, num_atom_features, num_bond_features,
        #                 args.radius, args.T, args.fingerprint_dim, args.p_dropout, task, return_emb)
        encoder, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        tokenizer = alphabet.get_batch_converter()
        num_layers, pro_hid_dim = 12, 768
        model = ESM1bPLI(encoder, num_layers, pro_hid_dim, num_atom_features, num_bond_features,
                         args.radius, args.T, args.fingerprint_dim, args.p_dropout, task, return_emb, random)
    elif model_name == 'prottrans':
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
        pro_hid_dim = model_hid_dims[model_name]
        model = ProtTransPLI(pro_hid_dim, num_atom_features, num_bond_features,
                            args.radius, args.T, args.fingerprint_dim, args.p_dropout, task, return_emb, random)
    elif model_name == 'tape':
        tokenizer = TAPETokenizer(vocab='iupac')
        pro_hid_dim = model_hid_dims[model_name]
        model = TAPEPLI(pro_hid_dim, num_atom_features, num_bond_features,
                        args.radius, args.T, args.fingerprint_dim, args.p_dropout, task, return_emb, random)

    if do_train:
        file_results = os.path.join(path_model, 'results.txt')
        file_loss = os.path.join(path_model, 'loss-metric.csv')
        f_results = open(file_results, 'a')

        start_time = time.time()
        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        trainer = Trainer(model_name, model, tokenizer, lr, weight_decay, batch_size, gradient_accumulation, return_emb, freeze_seq_encoder)
        tester = Tester(model_name, model, tokenizer, batch_size, return_emb, training=True)
        
        task_metric = task_metrics[task]
        results = (f'Epoch\tTime\tLoss_train\tLoss_dev\t{task_metric}_train\t{task_metric}_dev')
        with open(file_results, 'w') as f:
            f.write(results + '\n')

        """Start training."""
        print('Training...')
        print(results)
        
        min_loss_dev = float('inf')
        max_metric_dev = -float('inf')
        best_epoch = 0

        loss_train_epochs, loss_dev_epochs = [], []
        metric_train_epochs, metric_dev_epochs = [], []
        for epoch in range(1, epochs+1):
            start_time_epoch = time.time()
            if epoch % decay_interval == 0:
                print('LR decay from {:.6f} to {:.6f}'.format(trainer.optimizer.param_groups[0]['lr'],
                                                              trainer.optimizer.param_groups[0]['lr']*lr_decay))
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss_train, all_predict_labels_train, all_real_labels_train = trainer.train(train_set, device, task)
            loss_dev, all_predict_labels_dev, all_real_labels_dev = tester.test(valid_set, device, task)

            if task == 'PDBBind':
                # import ipdb; ipdb.set_trace()
                metric_train, _ = pearsonr(all_real_labels_train, all_predict_labels_train)
                metric_dev, _ = pearsonr(all_real_labels_dev, all_predict_labels_dev)
            elif task in ['Kinase', 'DUDE', 'GPCR']:
                metric_train = roc_auc_score(all_real_labels_train, all_predict_labels_train) # stats.prc
                metric_dev = roc_auc_score(all_real_labels_dev, all_predict_labels_dev) # stats.prc

            loss_train_epochs.append(float("%.3f" % loss_train)), loss_dev_epochs.append(float("%.3f" % loss_dev))
            metric_train_epochs.append(float("%.3f" % metric_train)), metric_dev_epochs.append(float("%.3f" % metric_dev))

            end_time_epoch = time.time()
            seconds = end_time_epoch-start_time_epoch
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)
            loss_train_epoch = "%.3f" % loss_train; loss_dev_epoch = "%.3f" % loss_dev
            metric_train_epoch = "%.3f" % metric_train; metric_dev_epoch = "%.3f" % metric_dev
            results = [epoch, spend_time_epoch, loss_train_epoch, loss_dev_epoch, metric_train_epoch, metric_dev_epoch]
            with open(file_results, 'a') as f:
                f.write('\t'.join(map(str, results)) + '\n')
            if metric_dev > max_metric_dev:
                min_loss_dev = loss_dev
                max_metric_dev = metric_dev
                best_epoch = epoch
                best_model = copy.deepcopy(model)
            print('\t'.join(map(str, results)))

        # 保存最佳模型
        tester.save_model(best_model, os.path.join(path_model, 'model-epoch_{}-metric_{:.3f}.pth'.format(best_epoch, max_metric_dev)))
        end_time = time.time()
        seconds = end_time-start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time = "%02d:%02d:%02d" % (h, m, s)

        dict_loss = {}
        dict_loss['epochs'] = list(range(1, epochs+1))
        dict_loss['loss_train_all'] = loss_train_epochs
        dict_loss['loss_dev_all'] = loss_dev_epochs
        dict_loss[f'{task_metric}_train_all'] = metric_train_epochs
        dict_loss[f'{task_metric}_dev_all'] = metric_dev_epochs

        df_loss = pd.DataFrame(dict_loss)
        df_loss.to_csv(file_loss, index=False)

        plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, path_model, 'MSE Loss', task)
        plot_train_dev_metric(list(range(1, epochs+1)), metric_train_epochs, metric_dev_epochs, path_model, task_metric, task)

        final_print = "All epochs spend %s, where the best model is in epoch %d" % (spend_time, best_epoch)
        print(final_print)
        f_results.write(final_print)
        if not do_test:
            f_results.close()

    if do_test:
        tester = Tester(model_name, best_model, tokenizer, batch_size, return_emb, training=True)
        loss_test, all_predict_labels_test, all_real_labels_test = tester.test(test_set, device, task)

        if task == 'PDBBind':
            metric_test, _ = pearsonr(all_real_labels_test, all_predict_labels_test)
        elif task in ['Kinase', 'DUDE', 'GPCR']:
            metric_test = roc_auc_score(all_real_labels_test, all_predict_labels_test)

        test_print = "Test results: Loss: %.3f, %s: %.3f" % (loss_test, task_metric, metric_test)
        print(test_print)
        f_results.write(test_print)
        f_results.close()

    if do_save_emb:
        if best_model is None:
            # 加载微调效果最好的模型
            suffix = '2023-03-29-02:15:46'
            best_model_state_dict = torch.load("outputs/model-2023-03-29-02:15:46/model_test/model--epoch-6.pth", map_location=torch.device('cpu'))
            model.load_state_dict(best_model_state_dict)
            best_model = model.to(device)
        else:
            suffix = current_time

        tester = Tester(model_name, best_model, tokenizer, batch_size, return_emb, training=False)
        print("Embedding dataset num: %d" % len(train_set.dataset))
        loss, all_predict_labels, all_real_labels, all_pro_ids, all_pro_seqs, all_pro_embs = tester.test(train_set, device, task)

        df_emb = pd.DataFrame()
        df_emb['pro_id'] = all_pro_ids
        df_emb['pro_seq'] = all_pro_seqs
        df_emb['pro_emb'] = all_pro_embs
        os.makedirs(f'embeddings/{model_name}', exist_ok=True)
        df_emb.to_pickle(f'embeddings/{model_name}/{task}_train_{model_name}_{suffix}.pkl')

    if do_save_pretrained_emb:
        best_model = model.to(device)
        tester = Tester(model_name, best_model, tokenizer, batch_size, return_emb, training=False)
        print("Embedding dataset num: %d" % len(train_set.dataset))
        loss, all_predict_labels, all_real_labels, all_pro_ids, all_pro_seqs, all_pro_embs = tester.test(train_set, device, task)

        df_emb = pd.DataFrame()
        df_emb['pro_id'] = all_pro_ids
        df_emb['pro_seq'] = all_pro_seqs
        df_emb['pro_emb'] = all_pro_embs
        os.makedirs(f'embeddings/{model_name}', exist_ok=True)
        df_emb.to_pickle(f'embeddings/{model_name}/{task}_train_{model_name}_without_finetune_{current_time}.pkl')
