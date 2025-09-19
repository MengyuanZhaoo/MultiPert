# MultiPert: An adversarially aligned and dual attention framework for single-cell multi-omics perturbation prediction

![Framework](https://github.com/user-attachments/assets/e10d07f0-b203-4c23-b154-afb182ab72eb)


# Overview
Precise prediction of perturbation responses is essential in systems biology research, as it plays a pivotal role in characterizing cellular identities and elucidating the regulatory mechanisms of biological pathways. Existing perturbation-responses prediction approaches are predominantly confined to single-modality transcriptomic data, limiting their capacity to capture cross-layer molecular effects. Here, we present MultiPert, a deep learning framework specifically designed for predicting perturbation responses in single-cell multi-omics data. MultiPert employs modality-specific encoders with dedicated pretraining, integrates perturbation through a dual-attention mechanism, and achieves cross-modal alignment via adversarial training. Benchmarking on human THP-1 and kidney multi-omics datasets demonstrates that MultiPert reliably predicts both perturbed gene expression and protein abundance profiles, achieving superior accuracy and stability compared to state-of-the-art strategies. MultiPert generalizes to unseen perturbations and uncovers regulatory mechanisms of immune checkpoint molecules based on perturbed proteomic predictions. In addition, enrichment analyzes of perturbed transcriptomic predictions reveal immune-related pathways. By providing an integrated and interpretable framework, MultiPert expands the scope of perturbation modeling at the multi-omics level, thereby offering a robust methodological foundation for comprehensive research into pathogenesis and drug discovery.
# Requirements
```Bash
numpy==1.23.5
pandas==2.0.2
scanpy==1.9.3
scikit-learn==1.2.2
matplotlib==3.7.1
scipy==1.12.0
torch==2.0.0
```
# Datasets
All data used in this study is publicly available. The THP-1 dataset was downloaded from [Zenodo](https://zenodo.org/records/7041849). The kidney dataset was downloaded from Gene Expression Omnibus (GEO) with accession number [GSE213957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213957).

# Usage
## Code Structure
- `data_loader.py` : Load and preprocess the dataset
- `utils.py` : Set random seeds
- `models.py` : Define the architecture of the whole network
- `trainer.py` : Define the process of model training and validation
- `metrics.py` : Define model evaluation-related functions
- `main.py` : Main function to run MultiPert model

## Detailed explanation of parameters
| Parameter | Type | Description |
| --- | --- | --- |
| `file_path` | `str` | Path to the directory containing the input data files. |
| `name` | `str` | Name of the dataset. |
| `out_path` | `str` | Path to the directory to save the output files. |
| `epochs` | `int` | Number of epochs to train the model. |

We have provided demo data in the dataset folder. After creating a virtual environment named env based on the requirements, users can accomplish MultiPert training and prediction with the following commands:
```Bash
source activate env
cd code
python main.py
```
All results are saved in the output folder by default.

If users apply MultiPert to a new dataset, the RNA.h5ad corresponding to transcriptome and protein.h5ad corresponding to proteome are required. Next, extract pert_embeddings.npz using the GEARS algorithm, referring to ./code/get_perturb.ipynb or [GEARS](https://github.com/snap-stanford/GEARS). Finally, run main.py to train MultiPert and perform multi-omics perturbation-response prediction.

# Comparison methods availability
| Method | Link |
| --- | --- |
| scGPT | https://github.com/bowang-lab/scGPT |
| scPRAM | https://github.com/jiang-q19/scPRAM |
| scGen | https://github.com/theislab/scgen |
| CoupleVAE | https://github.com/LiminLi-xjtu/CoupleVAE |
| CPA | https://github.com/theislab/cpa |
| GEARS | https://github.com/snap-stanford/GEARS |

