import json
import ast
import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, classification_report


class PPI_GDA_Data(InMemoryDataset):
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

    def __init__(self, G, labels, attrib, num_classes):
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

        super(PPI_GDA_Data, self).__init__('.', None, None, None)

        data = from_networkx(G, attrib)
        y = torch.from_numpy(labels).type(torch.long)

        data.x = data.x.float()
        data.y = y.clone().detach()
        data.num_classes = num_classes

        indices = range(G.number_of_nodes())
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(data.x, data.y, indices, test_size=0.3, stratify=labels, random_state=42)

        X_test, X_val, y_test, y_val, test_idx, val_idx = train_test_split(X_test, y_test, test_idx, test_size=0.5, stratify=y_test, random_state=42)


        n_nodes = G.number_of_nodes()
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
        for idx in train_idx:
            train_mask[idx] = True
        for idx in test_idx:
            test_mask[idx] = True
        for idx in val_idx:
            val_mask[idx] = True

        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        data['val_mask'] = val_mask

        self.data, self.slices = self.collate([data])


def getData(configFile, graph_file):
    """
    This function reads in the BioGRID (with DisGeNet features) data graph and
    APU scores to create an object of the PPI_GDA_Data class.
    
    WARNING: This function is not yet complete - does not handle the NOT 
        Quantile logic!

    Parameter
    ---------
    configFile : str
        This string provides the path to the configuration .json file.
    graph_file : str
        This string provides the filename of the .gml file, from which to load
        the PPI graph!

    Raises
    ------
    None
        None

    Returns
    -------
    tuple
        This function returns a tuple of (PPI_GDA_Data object, datagraph obj)

    """
    
    #HACK: duplicate effort of loading configuration json file
    with open(configFile, "r") as cfg:
        config = json.load(cfg)

    #FIXME: Hard-coded path
    nxG = nx.read_gml(config['projRootPath']+config['dataGraph']+graph_file)      

    #FIXME: Hard-coded path
    seed_genes = pd.read_csv(
        config['projRootPath']+'/data/seed_genes/C0006142_Malignant_neoplasm_of_breast_all_seed_genes.txt',
        header=None,
        sep=' '
    )
    seed_genes.columns = ["name", "GDA Score"]
    seeds_list = seed_genes["name"].values.tolist()

    #FIXME: Hard-coded path
    nedbit_scores = pd.read_csv(config["projRootPath"] + "/data/APU_scores/C0006142_Malignant_neoplasm_of_breast_features_Score.csv")

    # Remove seed genes
    nedbit_scores_not_seed = nedbit_scores[~nedbit_scores['name'].isin(seeds_list)]
    #print(nedbit_scores_not_seed.shape)

    # Sort scores for quartile division
    #NOTE: Why was this done? go figure! or ask authors of NIAPU and XGDAG
    nedbit_scores_not_seed = nedbit_scores_not_seed.sort_values(by = "out", ascending = False)
    pseudo_labels = pd.qcut(
        x=nedbit_scores_not_seed["out"],
        q=4,
        labels=["RN", "LN", "WN", "LP"]
    )
    
    nedbit_scores_not_seed['label'] = pseudo_labels

    nedbit_scores_seed = nedbit_scores[nedbit_scores['name'].isin(seeds_list)]
    nedbit_scores_seed = nedbit_scores_seed.assign(label = 'P')

    # Convert dataframe to dict for searching nodes and their labels
    not_seed_labels = dict(zip(nedbit_scores_not_seed['name'], nedbit_scores_not_seed['label']))
    seed_labels = dict(zip(nedbit_scores_seed['name'], nedbit_scores_seed['label']))

    labels_dict = {'P':0, 'LP': 1, 'WN': 2, 'LN': 3, 'RN': 4}
    labels = []

    for node in nxG:
        if node in not_seed_labels:
            labels.append(labels_dict[not_seed_labels[node]])
        else:
            labels.append(labels_dict[seed_labels[node]])

    labels = np.asarray(labels)
    
    attributes = ['degree', 'ring', 'NetRank', 'NetShort', 'HeatDiff', 'InfoDiff']

    #HACK: Is this the correct way to store and load a list from json?
    classes = ast.literal_eval(config['classList'])

    dataObj = PPI_GDA_Data(nxG, labels, attributes, len(classes))

    return (dataObj[0], nxG)

