# MultiPert
An adversarially aligned and attention-based framework for single-cell multi-omics perturbation prediction (MultiPert)

# Overview

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



