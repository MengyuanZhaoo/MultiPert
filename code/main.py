import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import scanpy as sc
import random
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from data_loader import *
from models import *
from trainer import train_and_validate
from metrics import *
from utils import set_seed


def main(file_path, name, out_path, epochs):
    set_seed(1)
    device = 'cpu'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         print(f"Using {torch.cuda.device_count()} GPUs!")
    #     else:
    #         print("Using 1 GPU!")
    # else:
    #     print("Using CPU!")
    

    file_rna = file_path+'/'+name+'_RNA.h5ad'
    file_adt = file_path+'/'+name+'_protein.h5ad'
    file_pert = file_path+'/pert_embeddings.npz'
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True) 

    # Loading and preprocessing data
    adata_rna_ctrl, adata_adt_ctrl, adata_rna_perturb, adata_adt_perturb, perturb_info, perturb_emb = load_and_preprocess_data(
        file_rna, file_adt, file_pert, out_path, device
    )
    rna_ctrl = adata_rna_ctrl.X.toarray()
    adt_ctrl = adata_adt_ctrl.X.toarray()
    rna_perturb = adata_rna_perturb.X.toarray()
    adt_perturb = adata_adt_perturb.X.toarray()
    name_rna = adata_rna_ctrl.var_names
    name_adt = adata_adt_ctrl.var_names
    cell_ctrl = np.array(adata_rna_ctrl.obs_names)
    cell_perturb = np.array(adata_rna_perturb.obs_names)

    rna_ctrl_sample = []
    adt_ctrl_sample = []
    cell_ctrl_sample = []
    unique_perturbs = np.unique(perturb_info)
    for perturb in unique_perturbs:
        perturb_mask = perturb_info == perturb
        rna_perturb_single = rna_perturb[perturb_mask]
        if len(rna_perturb_single) > 0:
            sampled_rna, sampled_adt, sampled_cell = sample_data(rna_ctrl, adt_ctrl, cell_ctrl, rna_perturb_single.shape[0])
            rna_ctrl_sample.append(sampled_rna)
            adt_ctrl_sample.append(sampled_adt)
            cell_ctrl_sample.append(sampled_cell)
    rna_ctrl_sample = np.concatenate(rna_ctrl_sample)
    adt_ctrl_sample = np.concatenate(adt_ctrl_sample)
    cell_ctrl_sample = np.concatenate(cell_ctrl_sample)

    # Datasets creation 
    batch_size = 512   
    dataset = MultiOmicsDataset(rna_ctrl_sample, adt_ctrl_sample,  perturb_info, rna_perturb, adt_perturb, perturb_emb, cell_ctrl_sample, cell_perturb)
    train_sampler, val_sampler, test_sampler = split_dataset(dataset, perturb_info)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Model initialization
    model_list = initialize_models(
        train_loader, rna_ctrl_sample.shape[1], adt_ctrl_sample.shape[1], perturb_emb.shape[1], device, out_path
    )

    # Model training
    best_model_exists = all([os.path.exists(os.path.join(out_path, f'model_best_{i}.pth')) for i in range(len(model_list))])
    if best_model_exists:
        print("Loading best saved models...")
        for i, model in enumerate(model_list):
            model.load_state_dict(torch.load(os.path.join(out_path, f'model_best_{i}.pth'), map_location=device))
    else:
        print("Integrated Training:")
        train_losses, val_losses, discriminator_losses, generator_losses = train_and_validate(model_list, train_loader, val_loader, device, out_path, epochs=epochs)

    # Model testing
    rna_perturb_test = []
    rna_perturb_pred = []
    adt_perturb_test = []
    adt_perturb_pred = []
    perturb_info_test = []
    rna_ctrl_test = []
    adt_ctrl_test = []
    cell_ctrl_test = []
    cell_perturb_test = []
    perturb_emb_test = []

    for model in model_list:
        model.eval()

    with torch.no_grad():
        for rna, adt, perturb_info_batch, perturbed_rna, perturbed_adt, perturb_emb, cell_ctrl_batch, cell_perturb_batch in test_loader:
            rna, adt, perturbed_rna, perturbed_adt, perturb_emb = rna.to(device), adt.to(device), perturbed_rna.to(
                device), perturbed_adt.to(device), perturb_emb.to(device)

            rna_ae, shared_encoder_rna, perturb_encoder_rna, decoder_rna, \
            adt_ae, shared_encoder_adt, perturb_encoder_adt, decoder_adt, \
            fusion_network, adversarial_network, attention_layer = model_list


            rna_specific, _ = rna_ae.encode(rna)
            rna_shared = shared_encoder_rna(rna)

            adt_specific = adt_ae.encoder(adt)
            adt_shared = shared_encoder_adt(adt)

            fused = fusion_network(rna_shared, adt_shared)

            rna_fused = torch.cat((rna_specific, fused), dim=1)
            perturbation_rna = perturb_encoder_rna(perturb_emb)

            rna_combined = attention_layer(rna_fused, perturbation_rna)
            final_rna = decoder_rna(rna_combined)

            adt_fused = torch.cat((adt_specific, fused), dim=1)
            perturbation_adt = perturb_encoder_adt(perturb_emb)

            adt_combined = attention_layer(adt_fused, perturbation_adt)
            final_adt = decoder_adt(adt_combined)

            rna_perturb_test.append(perturbed_rna.cpu().numpy())
            rna_perturb_pred.append(final_rna.cpu().numpy())
            adt_perturb_test.append(perturbed_adt.cpu().numpy())
            adt_perturb_pred.append(final_adt.cpu().numpy())
            perturb_info_test.extend(perturb_info_batch)
            rna_ctrl_test.extend(rna.cpu().numpy())
            adt_ctrl_test.extend(adt.cpu().numpy())
            cell_ctrl_test.extend(cell_ctrl_batch)
            cell_perturb_test.extend(cell_perturb_batch)

    rna_perturb_test = np.concatenate(rna_perturb_test)
    rna_perturb_pred = np.concatenate(rna_perturb_pred)
    adt_perturb_test = np.concatenate(adt_perturb_test)
    adt_perturb_pred = np.concatenate(adt_perturb_pred)
    perturb_info_test = np.array(perturb_info_test)
    rna_ctrl_test = np.array(rna_ctrl_test)
    adt_ctrl_test = np.array(adt_ctrl_test)
    cell_ctrl_test = np.array(cell_ctrl_test)
    cell_perturb_test = np.array(cell_perturb_test)

    # File saving
    save_to_h5ad(rna_ctrl_test, adt_ctrl_test, perturb_info_test, cell_ctrl_test, name_rna, out_path+'/test_control.h5ad')
    save_to_h5ad(rna_perturb_test, adt_perturb_test, perturb_info_test, cell_perturb_test, name_rna, out_path+'/test_perturb.h5ad')
    print(f"Test data saved to {out_path}")
    save_to_h5ad(rna_perturb_pred, adt_perturb_pred, perturb_info_test, cell_perturb_test, name_rna, out_path+'/predict.h5ad')
    print(f"Prediction data saved to {out_path}")

    # Model evaluation
    per_perturb_metrics = calculate_metrics(rna_perturb_test, rna_perturb_pred, perturb_info_test)
    per_perturb_de_metrics = calculate_de_metrics(rna_perturb_test, rna_perturb_pred, perturb_info_test, rna_ctrl_test, 50)

    metrics = {}
    rownames = ['MSE', 'MAE', 'PCC']
    rownames.extend([''])
    rownames.extend(['DEGs MSE', 'DEGs MAE','DEGs PCC'])
    for perturb in unique_perturbs:
        perturb_metrics = per_perturb_metrics[perturb]['expression']
        perturb_metrics.extend([np.nan])
        perturb_metrics.extend(per_perturb_de_metrics[perturb]['expression'])
        metrics[perturb] = perturb_metrics
    df_metrics = pd.DataFrame(metrics, index=rownames, columns=unique_perturbs)
    df_metrics.to_csv(out_path+ '/metrics_rna.csv', index=True)
    print('Metrics saved to '+ out_path+ '/metrics_rna.csv')

    per_perturb_metrics = calculate_metrics(adt_perturb_test, adt_perturb_pred, perturb_info_test)
    metrics = {}
    rownames = ['MSE', 'MAE', 'PCC']
    for perturb in unique_perturbs:
        perturb_metrics = per_perturb_metrics[perturb]['expression']
        metrics[perturb] = perturb_metrics
    df_metrics = pd.DataFrame(metrics, index=rownames, columns=unique_perturbs)
    df_metrics.to_csv(out_path+ '/metrics_adt.csv', index=True)
    print('Metrics saved to '+ out_path+ '/metrics_adt.csv')

if __name__ == "__main__":
    # For PapalexiSatija2021
    file_path = '../data'
    name = 'PapalexiSatija2021_eccite_arrayed'
    out_path = '../output'
    epochs = 1000

    main(file_path, name, out_path, epochs)
