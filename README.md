## Workshop on Gene-Disease Association based Gene Prioritization

This project aims at implementing the XGDAG Paper. The primary focus is to organize the implementation in as many logical blocks as possible, while recreating the results.

[TOC]

### The file structure of this project

```console
.
├── data
│   ├── all_gene_disease_associations.tsv
│   ├── APU_scores
│   │   ├── C0006142_Malignant_neoplasm_of_breast_features_Score.csv
│   │   ├── C0009402_Colorectal_Carcinoma_features_Score.csv
│   │   ├── C0023893_Liver_Cirrhosis_Experimental_features_Score.csv
│   │   ├── C0036341_Schizophrenia_features_Score.csv
│   │   ├── C0376358_Malignant_neoplasm_of_prostate_features_Score.csv
│   │   └── README.md
│   ├── APU_scores.zip
│   ├── BIOGRID-ORGANISM-4.4.231.tab3.zip
│   ├── BIOGRID-ORGANISM-Homo_sapiens-4.4.206.tab3.txt
│   ├── BIOGRID-ORGANISM-Homo_sapiens-4.4.231.tab3.txt
│   ├── DataGraphs
│   │   ├── dev_graphs
│   │   │   ├── grafo_prova_v2.gml
│   │   │   ├── graph_with_centrality.gml
│   │   │   ├── graph_with_non_normalized_nedbit.gml
│   │   │   ├── graph_with_normalized_centrality.gml
│   │   │   ├── graph_with_normalized_nedbit.gml
│   │   │   ├── graph_with_weighted_nedbit.gml
│   │   │   └── original_graph.gml
│   │   ├── grafo_diamond_nedbit_C0006142.gml
│   │   ├── grafo_diamond_nedbit_C0009402.gml
│   │   ├── grafo_diamond_nedbit_C0023893.gml
│   │   ├── grafo_nedbit_C0001973.gml
│   │   ├── grafo_nedbit_C0005586.gml
│   │   ├── grafo_nedbit_C0006142.gml
│   │   ├── grafo_nedbit_C0009402.gml
│   │   ├── grafo_nedbit_C0011581.gml
│   │   ├── grafo_nedbit_C0023893.gml
│   │   ├── grafo_nedbit_C0036341.gml
│   │   ├── grafo_nedbit_C0376358.gml
│   │   ├── grafo_nedbit_C0860207.gml
│   │   ├── grafo_nedbit_C3714756.gml
│   │   └── README.md
│   ├── DataGraphs.zip
│   ├── disease_associations.tsv
│   ├── disease_associations.tsv.gz
│   ├── disgenet_2020.db
│   ├── disgenet_2020.db.gz
│   ├── gda.tsv
│   ├── gene_associations.tsv
│   ├── gene_associations.tsv.gz
│   ├── Graphs
│   │   ├── bioGRID_graph.gml
│   │   ├── C0006142_nedbit.gml
│   │   └── README.md
│   ├── Models
│   │   ├── GNN7L_SAGEConv_57192.pt
│   │   └── README.md
│   ├── NeDBIT_features
│   │   ├── C0001973_Alcoholic_Intoxication_Chronic_features
│   │   ├── C0005586_Bipolar_Disorder_features
│   │   ├── C0006142_Malignant_neoplasm_of_breast_features
│   │   ├── C0009402_Colorectal_Carcinoma_features
│   │   ├── C0011581_Depressive_disorder_features
│   │   ├── C0023893_Liver_Cirrhosis_Experimental_features
│   │   ├── C0036341_Schizophrenia_features
│   │   ├── C0376358_Malignant_neoplasm_of_prostate_features
│   │   ├── C0860207_Drug_Induced_Liver_Disease_features
│   │   ├── C3714756_Intellectual_Disability_features
│   │   └── README.md
│   ├── NeDBIT_features.zip
│   ├── Rankings
│   │   ├── C0006142_ranks_firstFullRun.csv
│   │   └── README.md
│   ├── README.md
│   ├── seed_genes
│   │   ├── C0001973_Alcoholic_Intoxication_Chronic_all_seed_genes.txt
│   │   ├── C0005586_Bipolar_Disorder_all_seed_genes.txt
│   │   ├── C0006142_Malignant_neoplasm_of_breast_all_seed_genes.txt
│   │   ├── C0009402_Colorectal_Carcinoma_all_seed_genes.txt
│   │   ├── C0011581_Depressive_disorder_all_seed_genes.txt
│   │   ├── C0023893_Liver_Cirrhosis_Experimental_all_seed_genes.txt
│   │   ├── C0036341_Schizophrenia_all_seed_genes.txt
│   │   ├── C0376358_Malignant_neoplasm_of_prostate_all_seed_genes.txt
│   │   ├── C0860207_Drug_Induced_Liver_Disease_all_seed_genes.txt
│   │   ├── C3714756_Intellectual_Disability_all_seed_genes.txt
│   │   └── README.md
│   ├── ten_seed_genes.zip
│   ├── variant_associations.tsv.gz
│   └── vdisease_associations.tsv.gz
├── docs
│   └── README.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── rough.ipynb
├── scripts
│   ├── configFiles
│   │   ├── preproc_config.json
│   │   ├── rank_config.json
│   │   └── train_config.json
│   ├── generate_ranks.py
│   ├── infer.py
│   ├── __init__.py
│   ├── lib
│   │   ├── GraphSVX
│   │   │   ├── src
│   │   │   │   ├── data.py
│   │   │   │   ├── eval_multiclass.py
│   │   │   │   ├── eval.py
│   │   │   │   ├── explainers.py
│   │   │   │   ├── gengraph.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models.py
│   │   │   │   ├── plots.py
│   │   │   │   ├── train.py
│   │   │   │   └── utils.py
│   │   │   └── utils
│   │   │       ├── featgen.py
│   │   │       ├── graph_utils.py
│   │   │       ├── __init__.py
│   │   │       ├── io_utils.py
│   │   │       ├── pipeline_figure.png
│   │   │       ├── synthetic_structsim.py
│   │   │       └── train_utils.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── model
│   │   ├── __init__.py
│   │   └── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils
│       ├── databaseElements.py
│       ├── dataclass.py
│       ├── generate_pipeline.py
│       ├── helpers.py
│       └── __init__.py
├── setup.cfg
├── setup.py
├── tests
│   └── test.py
└── tox.ini
```

### Understanding the project

The starting point of this project is `scripts/preprocess.py`. After executing
it, all necessary data files will be generated under the `data` directory.



### Documentation Strings in Python code

The code files in this project use the following docstring style:

```Python
class AClass:
    """
    <Purpose>

    Attributes
    ----------
    atr1 : dtype
        <purpose>

    Methods
    -------
    __init__():
        Constructor
    method1(param):
        <purpose>
    
    """
    atr1 = 'False'
    
    def __init__(self):
        super(TSVParser, self).__init__()

    def method1(param):


def AFunction():
    """
    <Purpose>

    Parameter
    ---------
    arg1 : dtype
        <purpose>

    Raises
    ------
    err1
        <purpose>

    Returns
    -------
    dtype
        <purpose>

    """
```
Further, the following `codetags` have been used throughout:
```Python
#BUG: 
#FIXME: 
#HACK: 
#NOTE: 
#TODO: 
```


### Bibliography
#### Gene-Disease Association
1. [XGDAG: explainable gene–disease associations via graph neural networks](https://academic.oup.com/bioinformatics/article/39/8/btad482/7235567)
2. (_extra_) [GNN-SubNet: disease subnetwork detection with explainable graph neural networks](https://academic.oup.com/bioinformatics/article/38/Supplement_2/ii120/6702000)

#### Graph Explainer
1. [GraphSVX: Shapley Value Explanations for Graph Neural Networks](https://arxiv.org/abs/2104.10482)
2. [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)

#### Data Processing
1. [NIAPU: network-informed adaptive positive-unlabeled learning for disease gene identification](https://arxiv.org/abs/2108.06158)
2. [XGDAG - supplementary data](https://academic.oup.com/bioinformatics/article/39/8/btad482/7235567#supplementary-data)

#### Feature Extraction
1. [Network propagation in the cytoscape cyberinfrastructure](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5638226/) (**Heat Diffusion Feature**)
2. [Network diffusion with centrality measures to identify disease-related genes](https://pubmed.ncbi.nlm.nih.gov/33892577/) (**Heat Diffusion Feature**)
3. [Algorithms for estimating relative importance in networks](https://dl.acm.org/doi/10.1145/956750.956782) (**NetShort**)
4. [Ring structures and mean first passage time in networks](https://pubmed.ncbi.nlm.nih.gov/16605394/) (**NetRing**)
