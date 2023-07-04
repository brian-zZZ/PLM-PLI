import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from AttentiveFP import save_smiles_dicts, get_smiles_array


class DataHandler:
    def __init__(self, raw_filename, args):
        self.args = args
        self.data_df, self.smile_feature_dict = self.load_smile(raw_filename)

    def load_smile(self, raw_filename):
        # raw_filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1.csv"
        feature_filename = raw_filename.replace('.csv', '.pickle')
        filename = raw_filename.replace('.csv', '')
        # smiles_tasks_df : df : ["unnamed", "PDB-ID", "seq", "SMILES", "rdkit_smiles", "Affinity-Value", "set"]
        smiles_tasks_df = pd.read_csv(raw_filename)  # main file
        # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
        smilesList = smiles_tasks_df[ self.args.SMILES].values
        print("number of all smiles: ", len(smilesList))
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject num from molecule obeject
                remained_smiles.append(smiles)  # list : smiles without transformation error
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))  # canonical smiles without transformation error
            except:
                print("the smile \"%s\" has transformation error in the first test" % smiles)
                pass
        print("number of successfully processed smiles after the first test: ", len(remained_smiles))

        "----------------------the first test----------------------"
        smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[ self.args.SMILES].isin(remained_smiles)]  # df(13464) : include smiles without transformation error
        smiles_tasks_df[ self.args.SMILES] = remained_smiles

        # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
        smilesList = remained_smiles  # update valid smile

        # feature_dicts(dict) : 
        # {smiles_to_atom_info, smiles_to_atom_mask, smiles_to_atom_neighbors, "smiles_to_bond_info", "smiles_to_bond_neighbors", "smiles_to_rdkit_list"}
        if os.path.isfile(feature_filename):  # get smile feature dict
        # if False:
            feature_dicts = pickle.load(open(feature_filename, "rb"))
        else:
            # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
            # filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1"
            feature_dicts = save_smiles_dicts(smilesList, filename)
        
        "----------------------the second test----------------------"
        # remained_df : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]) : include smiles without transformation error and second test error, 13435
        remained_df = smiles_tasks_df[smiles_tasks_df[ self.args.SMILES].isin(feature_dicts['smiles_to_atom_mask'].keys())]
        print("number of successfully processed smiles after the second test: ", len(remained_df))

        return remained_df, feature_dicts


class ProteinDataset(Dataset):
    def __init__(self, dataset, data_handler, args):
        super(ProteinDataset, self).__init__()
        self.dataset = dataset
        self.data_handler = data_handler
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data_entry = self.dataset.iloc[item]
        smiles_list = [data_entry[self.args.SMILES]]
        y_val = data_entry[self.args.TASK]
        y_val = torch.tensor(y_val)
        pro_seq = data_entry.seq
        pro_seq = pro_seq if len(pro_seq)<=self.args.max_seq_len else pro_seq[:self.args.max_seq_len]
        pro_id = data_entry[self.args.ID]
        
        # Generate seq, drug inputs
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
                                                            smiles_list, self.data_handler.smile_feature_dict)
        # print(tokenized_sent.shape, [e.shape for e in [amino_list, amino_degree_list, amino_mask]])
        return y_val, pro_seq, pro_id, (x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

    
def prepare_data(args, task, sampled_frac=1):
    data_handler = DataHandler(args.input_files_dict[task], args)
    if task == 'PDBBind':
        train_df = data_handler.data_df[data_handler.data_df["set"].str.contains('train')].reset_index(drop=True)#.iloc[:50]#
        valid_df = data_handler.data_df[data_handler.data_df["set"].str.contains('valid')].reset_index(drop=True)#.iloc[:20] #.iloc[:20]
        test_df = data_handler.data_df[data_handler.data_df["set"].str.contains('test')].reset_index(drop=True)#.iloc[:10] #.iloc[:10]
        test_casf2013_df = data_handler.data_df[data_handler.data_df["set"].str.contains('casf2013')].reset_index(drop=True) #.iloc[:10]
        test_astex_df = data_handler.data_df[data_handler.data_df["set"].str.contains('astex')].reset_index(drop=True) #.iloc[:10]
        if sampled_frac < 1: # 采样
            train_df = train_df.sample(frac=sampled_frac, random_state=args.SEED)
        # train_df_nums: 11000, valid_df_nums: 914, core2016_df_nums: 274, casf2013_df_nums: 180, astex_df_nums: 70
        print("train_df_nums: %d, valid_df_nums: %d, core2016_df_nums: %d, casf2013_df_nums: %d, astex_df_nums: %d" 
            % (len(train_df), len(valid_df), len(test_df), len(test_casf2013_df), len(test_astex_df)))

        train_set = ProteinDataset(train_df, data_handler, args)
        valid_set = ProteinDataset(valid_df, data_handler, args)
        test_set = ProteinDataset(test_df, data_handler, args)
        test_casf2013_set = ProteinDataset(test_casf2013_df, data_handler, args)
        test_astex_set = ProteinDataset(test_astex_df, data_handler, args)

        dataset_pack = train_set, valid_set, test_set, test_casf2013_set, test_astex_set
    elif task in ['Kinase', 'DUDE', 'GPCR']:
        train_df = data_handler.data_df[data_handler.data_df["set"].str.contains('train')].reset_index(drop=True)#.iloc[:500]
        test_df = data_handler.data_df[data_handler.data_df["set"].str.contains('test')].reset_index(drop=True) #.iloc[:10]
        if sampled_frac < 1:
            train_df = train_df.sample(frac=sampled_frac, random_state=args.SEED)
            test_df = test_df.sample(frac=sampled_frac, random_state=args.SEED)
        print("train_df_nums: %d, test_df_nums: %d" % (len(train_df), len(test_df)))
        train_set = ProteinDataset(train_df, data_handler, args)
        test_set = ProteinDataset(test_df, data_handler, args)
        dataset_pack = train_set, test_set
    else:
        raise ValueError
    
    x_atom, x_bonds, _, _, _, _ = get_smiles_array([data_handler.data_df[args.SMILES][1]], data_handler.smile_feature_dict)
    num_atom_features = x_atom.shape[-1]  # 39
    num_bond_features = x_bonds.shape[-1]  # 10

    return dataset_pack, num_atom_features, num_bond_features
