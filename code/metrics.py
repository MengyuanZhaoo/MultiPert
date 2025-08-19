import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def calculate_base_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    return mse, mae, pcc


def select_top_k_de_genes(data, perturb_info, rna_control, k=50):
    top_k_de_genes = {}
    unique_perturbs = np.unique(perturb_info)
    for perturb in unique_perturbs:
        perturb_mask = perturb_info == perturb
        perturb_data = data[perturb_mask]

        matched_control = rna_control[perturb_mask]
        combined_rna = np.vstack([matched_control, perturb_data])
        combined_perturb_info = np.concatenate([['control'] * len(matched_control), [perturb] * len(perturb_data)])
        adata = sc.AnnData(combined_rna)
        adata.obs['perturb'] = combined_perturb_info
        sc.tl.rank_genes_groups(adata, groupby='perturb', reference='control', method='wilcoxon')
        top_k_de_genes[perturb] = adata.uns['rank_genes_groups']['names'][perturb][:k]
    return top_k_de_genes

def calculate_de_metrics(all_perturbed_rna_test, all_final_rna_test, all_perturb_info_test, rna_control, k=50):
    test_top_k_de_genes = select_top_k_de_genes(all_perturbed_rna_test, all_perturb_info_test, rna_control, k)
    pred_top_k_de_genes = select_top_k_de_genes(all_final_rna_test, all_perturb_info_test, rna_control, k)

    per_perturb_de_metrics = {}
    for perturb in test_top_k_de_genes.keys():
        perturb_mask = all_perturb_info_test == perturb
        test_expr = all_perturbed_rna_test[perturb_mask]
        pred_expr = all_final_rna_test[perturb_mask]

        test_gene_indices = [int(gene) for gene in test_top_k_de_genes[perturb]]
        test_expr = test_expr[:, test_gene_indices]
        pred_expr = pred_expr[:, test_gene_indices]

        expr_mse, expr_mae, expr_pcc = calculate_base_metrics(test_expr.flatten(), pred_expr.flatten())

        per_perturb_de_metrics[perturb] = {
            'expression': [expr_mse, expr_mae, expr_pcc]
        }
    return per_perturb_de_metrics

def calculate_metrics(all_perturbed_rna_test, all_final_rna_test, all_perturb_info_test):
    per_perturb_metrics = {}
    unique_perturbs = np.unique(all_perturb_info_test)
    for perturb in unique_perturbs:

        perturb_mask = all_perturb_info_test == perturb
        test_expr = all_perturbed_rna_test[perturb_mask]
        pred_expr = all_final_rna_test[perturb_mask]

        expr_mse, expr_mae, expr_pcc = calculate_base_metrics(test_expr, pred_expr)

        per_perturb_metrics[perturb] = {
            'expression': [expr_mse, expr_mae, expr_pcc]
        }
    return per_perturb_metrics
