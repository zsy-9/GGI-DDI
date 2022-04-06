from rdkit import Chem
import pandas as pd
import numpy as np
import pickle
import os
import torch
from tqdm import tqdm

#1、生成药物的初始表示文件
#1.1、判断原子是否具有某种属性
def one_of_k_encoding(k, possible_values):#k和possible_values中每个元素是否相等的list
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):#x和allowable_set中每个元素是否相等的list
    if x not in allowable_set:
        x = allowable_set[-1]#-1：最后一个元素
    return list(map(lambda s: x == s, allowable_set))
#1.2、为原子生成特征向量
def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    results = np.array(results).astype(np.float32)
    return torch.from_numpy(results)
#1.3、为原子间连边生成特征向量
def edge_features(bond):#获取连边的初始表示
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()
#1.4、生成药物分子的表示
def generate_drug_data(mol_graph, atom_symbols):
    #连边：
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    # edge_list：b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    # edge_feats：edge_features(b)
    new_edge_index = edge_list.T
    #分子
    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()#根据Idx排序
    _, features = zip(*features)#把Idx去掉了
    features = torch.stack(features)#特征组成一个向量
    return features, new_edge_index, edge_feats

#1.5、保存文件
def save_data(data, filename):
    dirname = f'D:/模型们/topk_GNNSUB'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')

#1.5、生成药物表示文件
def load_drug_mol_data():
    data = pd.read_excel('D:/模型们/topk_GNNSUB/DDI_data.xlsx')
    drug_id_mol = []
    symbols = list()
    drug_smile_dict = {}
    for i in range(len(data)):
        id1, id2, smiles1, smiles2, interaction ,label = data.loc[i].values
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2
    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols) for id, mol in tqdm(drug_id_mol, desc='Processing drugs')}
    save_data(drug_data, 'drug_data.pkl')
    return drug_data




