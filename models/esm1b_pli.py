import os
import numpy as np
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentiveFP import Fingerprint


class ESM1b(nn.Module):
    def __init__(self, encoder, num_layers, pro_hid_dim=1280, random=False):
        super().__init__()
        self.encoder = encoder
        self.num_layers = num_layers
        if random:
            # 重新随机初始化权重
            print("Re-init pretrained model weights")
            for name, m in self.encoder.named_parameters():
                if 'weight' in name:
                    if m.ndim >= 2:
                        torch.nn.init.xavier_uniform_(m)
                    else:
                        torch.nn.init.uniform_(m)
                if 'bias' in name:
                    torch.nn.init.zeros_(m)

    def make_masks(self, proteins_num):
        N = len(proteins_num)  # batch size
        protein_max_len = torch.max(proteins_num)
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            protein_mask[i, :proteins_num[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.to(proteins_num.device)
        return protein_mask

    def forward(self, protein_data):
        pro_ids, seq_feat, proteins_num = protein_data

        proteins_mask = self.make_masks(proteins_num)
        pro_feat = self.encoder(seq_feat, repr_layers=[self.num_layers], return_contacts=True)
        pro_feat = pro_feat["representations"][self.num_layers]
        pro_feat = pro_feat.squeeze(0)  # mark esm_msa_1b时保留
        norm = torch.norm(pro_feat, dim=2)

        proteins_mask = proteins_mask.squeeze(-2).squeeze(-2)
        norm = norm.masked_fill(proteins_mask==0, -1e10)

        norm = F.softmax(norm, dim=1)
        norm = norm.unsqueeze(-1)
        pro_feat = pro_feat * norm      
        
        return pro_feat
    

class ESM1bPLI(nn.Module):
    def __init__(self, encoder, num_layers, pro_hid_dim, input_feat_dim, input_bond_dim, radius, T, fingerprint_dim, p_dropout, task, return_emb=False, random=False):
        super(ESM1bPLI, self).__init__()
        # ESM1b编码蛋白质序列特征
        self.esm1b_encoder = ESM1b(encoder, num_layers, pro_hid_dim, random)
        # GAT编码药物特征
        self.gat_encoder = Fingerprint(radius, T, input_feat_dim, input_bond_dim, fingerprint_dim, p_dropout)
        # 预测. fc重投影
        self.fc1 = nn.Sequential(nn.Dropout(p_dropout),
                                 nn.Linear(pro_hid_dim+fingerprint_dim, pro_hid_dim))
        if task == 'PDBBind': # 回归任务
            self.fc2 = nn.Sequential(nn.ReLU(),
                                     nn.Dropout(p_dropout),
                                     nn.Linear(pro_hid_dim, 1))
        elif task in ['Kinase', 'DUDE', 'GPCR']: # 二分类任务
            self.fc2 = nn.Sequential(nn.ReLU(),
                                     nn.Dropout(p_dropout),
                                     nn.Linear(pro_hid_dim, 2))
        self.return_emb = return_emb

    def forward(self, protein_data, drug_data):
        protein_emb = self.esm1b_encoder(protein_data)
        # import ipdb; ipdb.set_trace()
        # protein_emb = protein_emb.squeeze(0)
        protein_emb = protein_emb.mean(dim=1)
        atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask = drug_data
        drug_emb = self.gat_encoder(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        pro_drug_emb = torch.cat((protein_emb, drug_emb), dim=1)
        pro_drug_emb = self.fc1(pro_drug_emb)
        predict = self.fc2(pro_drug_emb)
        if not self.return_emb:
            return predict
        else:
            return predict, pro_drug_emb
    

class ESM1bPLI0(nn.Module):
    def __init__(self, encoder, num_layers, pro_hid_dim, input_feat_dim, input_bond_dim, radius, T, fingerprint_dim, p_dropout, task, return_emb=False):
        super(ESM1bPLI0, self).__init__()
        # ESM1b编码蛋白质序列特征
        self.esm1b_encoder = ESM1b(encoder, num_layers, pro_hid_dim)
        # GAT编码药物特征
        self.gat_encoder = Fingerprint(radius, T, input_feat_dim, input_bond_dim, fingerprint_dim, p_dropout)
        # 预测. fc重投影?
        if task == 'PDBBind': # 回归任务
            self.predictor = nn.Sequential(nn.Dropout(p_dropout),
                                            nn.Linear(pro_hid_dim+fingerprint_dim, 1))
        elif task in ['Kinase', 'DUDE', 'GPCR']: # 二分类任务
            self.predictor = nn.Sequential(nn.Dropout(p_dropout),
                                            nn.Linear(pro_hid_dim+fingerprint_dim, 2))
        self.return_emb = return_emb

    def forward(self, protein_data, drug_data):
        protein_emb = self.esm1b_encoder(protein_data)
        # import ipdb; ipdb.set_trace()
        # protein_emb = protein_emb.squeeze(0)
        protein_emb = protein_emb.mean(dim=1)
        atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask = drug_data
        drug_emb = self.gat_encoder(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        pro_drug_emb = torch.cat((protein_emb, drug_emb), dim=1)
        predict = self.predictor(F.relu(pro_drug_emb))
        if not self.return_emb:
            return predict
        else:
            return predict, pro_drug_emb
