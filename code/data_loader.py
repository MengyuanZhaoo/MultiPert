import torch
import os
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def preprocess_transcriptomics(data, n_hvgs):
    sc.pp.filter_genes(data, min_cells = 10)
    sc.pp.normalize_total(data, 10000)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data,  n_top_genes=n_hvgs)
    data = data[:, data.var['highly_variable']]
    return data

def preprocess_proteomics(data):
    sc.pp.normalize_total(data, 10000)
    sc.pp.log1p(data)
    return data

def remove_rare_perturbations(rna_adata, adt_adata, perturbation_column='perturbation', min_cell_count=10):
    perturbation_counts = rna_adata.obs[perturbation_column].value_counts()
    rare_perturbations = perturbation_counts[perturbation_counts < min_cell_count].index
    keep_cells = ~rna_adata.obs[perturbation_column].isin(rare_perturbations)
    rna_adata = rna_adata[keep_cells, :]
    adt_adata = adt_adata[keep_cells, :]
    return rna_adata, adt_adata

def load_pert_embeddings(file_path):
    data = np.load(file_path)
    return {name: embedding for name, embedding in zip(data['pert_list'], data['pert_embeddings'])}

def read_h5ad(file_path):
    adata = sc.read_h5ad(file_path)
    data = adata.X.toarray()
    perturb_info = adata.obs['perturbation'].values
    return data, perturb_info

class MultiOmicsDataset(Dataset):
    def __init__(self, rna_data, adt_data, perturb_info, perturbed_rna, perturbed_adt, perturb_emb, cell_ctrl_sample, cell_perturb):
        self.rna_data = rna_data.astype(np.float32)
        self.adt_data = adt_data.astype(np.float32)
        self.perturb_info = perturb_info
        self.perturbed_rna = perturbed_rna.astype(np.float32)
        self.perturbed_adt = perturbed_adt.astype(np.float32)
        self.perturb_emb = perturb_emb
        self.cell_ctrl_sample = cell_ctrl_sample
        self.cell_perturb = cell_perturb

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        return self.rna_data[idx], self.adt_data[idx], self.perturb_info[idx], \
               self.perturbed_rna[idx], self.perturbed_adt[idx], self.perturb_emb[idx], \
               self.cell_ctrl_sample[idx], self.cell_perturb[idx]

def split_dataset(dataset, perturb_info, train_ratio=0.6, val_ratio=0.2):
    unique_perturbs = np.unique(perturb_info)
    train_indices = []
    val_indices = []
    test_indices = []
    for perturb in unique_perturbs:
        perturb_indices = np.where(perturb_info == perturb)[0]
        train_size = int(train_ratio * len(perturb_indices))
        val_size = int(val_ratio * len(perturb_indices))
        test_size = len(perturb_indices) - train_size - val_size

        np.random.shuffle(perturb_indices)
        train_indices.extend(perturb_indices[:train_size])
        val_indices.extend(perturb_indices[train_size:train_size + val_size])
        test_indices.extend(perturb_indices[train_size + val_size:])
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, val_sampler, test_sampler

def sample_data(rna_data, adt_data, cell_ctrl, num_perturbed_cells):
    if rna_data.shape[0] >= num_perturbed_cells:
        indices = np.random.choice(rna_data.shape[0], num_perturbed_cells, replace=False)
    else:
        indices = np.random.choice(rna_data.shape[0], num_perturbed_cells, replace=True)
    sampled_rna = rna_data[indices]
    sampled_adt = adt_data[indices]
    sampled_cell = cell_ctrl[indices]
    return sampled_rna, sampled_adt, sampled_cell

def save_to_h5ad(rna_data, adt_data, perturb_info, cells, features, file_path):
    adata = sc.AnnData(X=rna_data)
    adata.obsm['adt'] = adt_data
    adata.obs = pd.DataFrame({'perturb': perturb_info},index=cells)
    adata.var = pd.DataFrame(index=features)
    adata.write_h5ad(file_path)

def load_and_preprocess_data(file_rna, file_adt, file_pert, out_path, device):
    if os.path.exists(out_path+'/preprocessed_rna.h5ad'):
        adata_rna_all = sc.read_h5ad(out_path+'/preprocessed_rna.h5ad')
        adata_adt_all = sc.read_h5ad(out_path+'/preprocessed_adt.h5ad')
        adata_rna_all.obs_names_make_unique()
        adata_adt_all.obs_names_make_unique()

        if 'cell_type' in adata_rna_all.obs.columns:
            adata_rna_all = adata_rna_all[adata_rna_all.obs['cell_type'].sort_values().index].copy()
            adata_adt_all = adata_adt_all[adata_adt_all.obs['cell_type'].sort_values().index].copy()
        print('Successfully loaded preprocessed data...')
    else:
        adata_rna_all = sc.read_h5ad(file_rna)
        adata_adt_all = sc.read_h5ad(file_adt)

        adata_rna_all, adata_adt_all = remove_rare_perturbations(adata_rna_all, adata_adt_all)
        adata_rna_all = preprocess_transcriptomics(adata_rna_all, 5000)
        adata_adt_all = preprocess_proteomics(adata_adt_all)
        assert adata_rna_all.obs.index.equals(adata_adt_all.obs.index)
        adata_rna_all.write_h5ad(out_path+'/preprocessed_rna.h5ad')
        adata_adt_all.write_h5ad(out_path+'/preprocessed_adt.h5ad')
        print(f"Preprocessed data saved to {out_path}")

    adata_rna_ctrl = adata_rna_all[adata_rna_all.obs['perturbation'] == 'control', :]
    adata_adt_ctrl = adata_adt_all[adata_adt_all.obs['perturbation'] == 'control', :]
    adata_rna_perturb = adata_rna_all[adata_rna_all.obs['perturbation'] != 'control', :]
    adata_adt_perturb = adata_adt_all[adata_adt_all.obs['perturbation'] != 'control', :]

    print("RNA Control:")
    print(adata_rna_ctrl)
    print("\nRNA Perturbed:")
    print(adata_rna_perturb)
    print("\nADT Control:")
    print(adata_adt_ctrl)
    print("\nADT Perturbed:")
    print(adata_adt_perturb)

    rna_perturbation = adata_rna_perturb.obs['perturbation'].astype(str)
    adt_perturbation = adata_adt_perturb.obs['perturbation'].astype(str)
    assert rna_perturbation.equals(adt_perturbation)
    perturb_info = rna_perturbation

    # Perturbation embedding
    unique_perturbs = np.unique(perturb_info)
    num_classes = len(unique_perturbs)
    perturb_mapping = {perturb: idx for idx, perturb in enumerate(unique_perturbs)}
    perturb_indices = np.array([perturb_mapping[perturb] for perturb in perturb_info])
    perturb_one_hot = np.eye(num_classes)[perturb_indices]
    perturb_emb = torch.from_numpy(perturb_one_hot).float().to(device)
    
    perturb_emb_go = load_pert_embeddings(file_pert)
    
    if 'TFRC' in perturb_emb_go:
        tfrc_embedding = perturb_emb_go.pop('TFRC')
        perturb_emb_go['CD71'] = tfrc_embedding
                    
    new_perturb_emb = []
    embedding_length = 64
    for i, perturb in enumerate(perturb_info):
        if perturb == 'PDL1':
            perturb = 'CD274'
        if perturb in perturb_emb_go.keys():
            embedding = perturb_emb_go[perturb]
            new_perturb_emb.append(torch.from_numpy(embedding).float())
        else:
            print(f"Perturbation {perturb} not found in embedding file, using one-hot instead.")
            one_hot = perturb_emb[i]
            extended_one_hot = torch.cat([one_hot, torch.zeros(embedding_length - len(one_hot))]).float()
            new_perturb_emb.append(extended_one_hot)
    perturb_emb = torch.stack(new_perturb_emb).to(device)
    
    return adata_rna_ctrl, adata_adt_ctrl, adata_rna_perturb, adata_adt_perturb, perturb_info, perturb_emb