import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import pandas as pd
import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from utils.pubchemfp import GetPubChemFPs

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_paths = ['./data/pretrain_data/MACCS_data/data10m_maccs', './data/pretrain_data/morgan_data/CHEMBL_morgan', './data/pretrain_data/PubChem_data/CHEMBL_pubChem']):
    smiles_list = []
    maccs_fp_list = []
    morgan_fp_list = []
    pubchem_fp_list = []
    for i in range(80):
        pretrain_data_0 = np.load(data_paths[0] + '_{}.npy'.format(i + 1), allow_pickle=True)
        pretrain_data_1 = np.load(data_paths[1] + '_{}.npy'.format(i + 1), allow_pickle=True)
        pretrain_data_2 = np.load(data_paths[2] + '_{}.npy'.format(i + 1), allow_pickle=True)
        if not(len(pretrain_data_0) == len(pretrain_data_1) and len(pretrain_data_0) == len(pretrain_data_2) and len(pretrain_data_1) == len(pretrain_data_2)):
            raise ValueError("not(len(pretrain_data_0) == len(pretrain_data_1) and len(pretrain_data_0) == len(pretrain_data_2) and len(pretrain_data_1) == len(pretrain_data_2)!")
        smiles_list = smiles_list + [x for x in pretrain_data_0[0]]
        maccs_fp_list = maccs_fp_list + [x for x in pretrain_data_0[4]]
        morgan_fp_list = morgan_fp_list + [x for x in pretrain_data_1[4]]
        pubchem_fp_list = pubchem_fp_list + [x for x in pretrain_data_2[4]]
        print('{}/80 of dataset'.format(i + 1) + ' is loaded')
    return smiles_list, maccs_fp_list, morgan_fp_list, pubchem_fp_list


"""
Remove a connected subgraph from the original molecule graph. 
Args:
    1. Original graph (networkx graph)
    2. Index of the starting atom from which the removal begins (int)
    3. Percentage of the number of atoms to be removed from original graph

Outputs:
    1. Resulting graph after subgraph removal (networkx graph)
    2. Indices of the removed atoms (list)
"""


def removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes) * percent))
    removed = []
    temp = [center]

    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        # self.smiles_data = read_smiles(data_path)

        self.smiles_data, self.maccs_fp, self.morgan_fp, self.pubchem_fp = read_smiles()

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        # Sample 2 different centers to start for i and j
        start_i, start_j = random.sample(list(range(N)), 2)

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        # Get the graph for i and j after removing subgraphs
        # G_i, removed_i = removeSubgraph(molGraph, start_i)
        # G_j, removed_j = removeSubgraph(molGraph, start_j)

        # percent_i, percent_j = random.uniform(0, 0.25), random.uniform(0, 0.25)
        percent_i, percent_j = 0.25, 0.25
        # percent_i, percent_j = 0.2, 0.2
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)

        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # x shape (N, 2) [type, chirality]

        # Mask the atoms in the removed list
        x_i = deepcopy(x)
        for atom_idx in removed_i:
            # Change atom type to 118, and chirality to 0
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        x_j = deepcopy(x)
        for atom_idx in removed_j:
            # Change atom type to 118, and chirality to 0
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])

        # Only consider bond still exist after removing subgraph
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)
            if (start, end) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)

        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

        maccs_fp = torch.tensor(self.maccs_fp[index], dtype=torch.int64)
        morgan_fp = torch.tensor(self.morgan_fp[index], dtype=torch.int64)
        pubchem_fp = torch.tensor(self.pubchem_fp[index], dtype=torch.int64)

        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp_maccs=maccs_fp, fp_morgan=morgan_fp, fp_pubchem=pubchem_fp)

        return data, data_i

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader


def build_maccs_pretrain_data_and_save(smiles_list, k100_list, k1000_list, k10000_list, save_path,
                                       global_feature='MACCS'):
    smiles_list = smiles_list
    k100_list = k100_list
    k1000_list = k1000_list
    k10000_list = k10000_list
    maccs_labels = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_list = np.array(maccs).tolist()
        # 选择负/正样本比例小于1000且大于0.001的数据
        selected_index = [3, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34,
                          36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                          59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                          104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                          123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
                          142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                          161, 162, 163, 164, 165]
        selected_macss_list = [maccs_list[x] for x in selected_index]
        maccs_labels.append(selected_macss_list)
        print(f'smiles: {i + 1}/{len(smiles_list)} to maccs')
    pretrain_data_list = [smiles_list, k100_list, k1000_list, k10000_list, maccs_labels]
    pretrain_data_np = np.array(pretrain_data_list, dtype=object)
    np.save(save_path, pretrain_data_np)


def build_pubchem_pretrain_data_and_save(smiles_list, k100_list, k1000_list, k10000_list, save_path,
                                         global_feature='PubChem'):
    smiles_list = smiles_list
    k100_list = k100_list
    k1000_list = k1000_list
    k10000_list = k10000_list
    pubChem_labels = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        pubChem = GetPubChemFPs(mol)
        pubChem_list = np.array(pubChem).tolist()
        pubChem_labels.append(pubChem_list)
        print(f'smiles: {i + 1}/{len(smiles_list)} to pubChem')
    pretrain_data_list = [smiles_list, k100_list, k1000_list, k10000_list, pubChem_labels]
    pretrain_data_np = np.array(pretrain_data_list, dtype=object)
    np.save(save_path, pretrain_data_np)


def build_erg_pretrain_data_and_save(smiles_list, k100_list, k1000_list, k10000_list, save_path, global_feature='erg'):
    smiles_list = smiles_list
    k100_list = k100_list
    k1000_list = k1000_list
    k10000_list = k10000_list
    phaErG_labels = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        phaErG_list = np.array(phaErGfp).tolist()
        phaErG_labels.append(phaErG_list)
        print(f'smiles: {i + 1}/{len(smiles_list)} to phaErG')
    pretrain_data_list = [smiles_list, k100_list, k1000_list, k10000_list, phaErG_labels]
    pretrain_data_np = np.array(pretrain_data_list, dtype=object)
    np.save(save_path, pretrain_data_np)


def build_morgan_pretrain_data_and_save(smiles_list, k100_list, k1000_list, k10000_list, save_path,
                                        global_feature='morgan'):
    smiles_list = smiles_list
    k100_list = k100_list
    k1000_list = k1000_list
    k10000_list = k10000_list
    morgan_labels = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        morganfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        morgan_list = np.array(morganfp).tolist()
        morgan_labels.append(morgan_list)
        print(f'smiles: {i + 1}/{len(smiles_list)} to morgan')
    pretrain_data_list = [smiles_list, k100_list, k1000_list, k10000_list, morgan_labels]
    pretrain_data_np = np.array(pretrain_data_list, dtype=object)
    np.save(save_path, pretrain_data_np)


if __name__ == "__main__":
    data_path = 'data/chem_dataset/zinc_standard_agent/processed/smiles.csv'
    # dataset = MoleculeDataset(data_path=data_path)
    # print(dataset)
    # print(dataset.__getitem__(0))
    dataset = MoleculeDatasetWrapper(batch_size=4, num_workers=4, valid_size=0.1, data_path=data_path)
    train_loader, valid_loader = dataset.get_data_loaders()
    for bn, (xis, xjs) in enumerate(train_loader):
        print(xis, xjs)
        break