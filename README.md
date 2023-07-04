# PLM-PLI
**Does protein pretrained language model facilitate the prediction of protein-ligand interaction?** \
A novel method that quantitatively assesses the significance of protein PLMs in PLI prediction

## Directory Structure
```bash
├── AttentiveFP/           # GAT model for extracting drug features
├── data/                  # PLI task datasets
├── models/                # PLMs
├── args.yaml              # Drug molecule parameters
├── config.py              # Configuration file for parameter settings
├── data_handler.py        # PLI data processing tool
├── main.py                # Main function
├── ot_metric              # Quantitative transfer metrics based on OT
├── OTFRM                  # OTFRM analysis
├── plotter.py             # Plotting tool
├── README.md              # Readme file
├── requirements.txt       # Environment dependencies
├── train_test.py          # Engine for training and testing the model
├── utils.py               # Collection of utility functions
```

## Requirements
[![python >3.10.11](https://img.shields.io/badge/python-3.10.11-brightgreen)](https://www.python.org/) [![torch-1.11.0](https://img.shields.io/badge/torch-1.11.0-orange)](https://github.com/pytorch/pytorch)

```bash
conda create -n PLMPLI python==3.10.11
conda activate PLMPLI
cd PLM-PLI
pip install -r requirements.txt
```

## Data Preparation
Place the processed datasets for PDBbind, Kinase, and DUD-E in the `data/` directory. An example entry of the processed PDBbind dataset is shown below:
|PDB-ID|seq|rdkit_smiles|label|set|
| :----- | :-----: | :-----: | :-----: | -----: |
|11gs|PYTVV...GKQ|CC[C@@H](CSC[C@H]...C(=O)c1ccc(OCC(=O)O)c(Cl)c1Cl|5.82|train|

## Fine-tuning on PLI Tasks
Run `main.py` to perform fine-tuning from pre-trained PLMs to downstream PLI prediction. The following example demonstrates the command to fine-tune using ProtTrans as the PLM on the PDBBind task:
```bash
python main.py --model_name=prottrans --task=PDBBind
```
For more input parameter settings, please refer to `config.py`.

## Acknowledgement
The SOFTWARE will be used for teaching or not-for-profit research purposes only. Permission is required for any commercial use of the Software.