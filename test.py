import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import torch
from utils import *

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

# mol smile to mol graph edge index

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))





def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # 创建一个大小为num_nodes x num_nodes的零矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    # 遍历edge index，将矩阵中对应的元素设置为1
    for edge in edge_index:
        node1, node2 = edge
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # 对于无向图，需要设置对称位置的值

    return adjacency_matrix




def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_tensor = edge_index_to_adjacency_matrix(edge_index, c_size)
    return c_size, features, edge_tensor

smile="CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1"
features, edge_tensor = smile_to_graph(smile)









# drugs = ["CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1","CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1"]
# compound_iso_smiles = drugs
# smile_graph = {}
# for smile in compound_iso_smiles:
#     g = smile_to_graph(smile)
#     smile_graph[smile] = g
#
# smile_graph = {}
# for smile in compound_iso_smiles:
#     g = smile_to_graph(smile)
#     smile_graph[smile] = g