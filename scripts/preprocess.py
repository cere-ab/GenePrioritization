import numpy as np
import pandas as pd
import json
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from utils.databaseElements import GeneDiseaseAssociation as GDA


class TSVParser:
    """
    The TSVParser class will take multiple tab separated csv files as input and generate processed data for both the GNN module and the post-processing module. The output data generated, will generally be in PyTorch tensor format.

    Attributes
    ----------
    None : None
        The TSVParser class does not have any class variable.

    Methods
    -------
    m(p):
        purpose
    
    """
    
    def __init__(self):
        """
        Purpose

        Parameter
        ---------
        None : None
            This function does not take any parameter.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return any value.

        """

        super(TSVParser, self).__init__()


class DataGraph:
    """
    The DataGraph class takes in necessary input data (BioGRID and DisGeNet) files and creates a graph of PPI with features extracted from the GDA.

    Attributes
    ----------
    None : None
        The DataGraph class does not have any class variable.

    Methods
    -------
    __init__(self, configFile):
        The constructor for the DataGraph class.
    buildGraph(self):
        Build the PPI graph and save to disk.
    prepareFeatures(self):
        Assign the nodes in the PPI graph, features, based on the GDA to be used in downstream tasks.
    
    """
    
    def __init__(self, configFile):
        """
        Constructor for the DataGraph class; loads the pre-processing configuration json file and declares some instance variables.

        Parameter
        ---------
        configFile : str
            The path to the pre-processing configuration json file.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return any value.

        """

        super(DataGraph, self).__init__()
        with open(configFile, "r") as cfg:
            self.config = json.load(cfg)
        self.biog = None
        self.interactomeGraph = None

    def buildGraph(self):
        """
        This function reads in the BioGRID data file (csv) and creates a graph from it - each gene is a node and the interaction between them is an edge. Some filtering is performed such as removal of self-loops and selection of largest connected component. Finally, the graph is stored on disk.

        Parameter
        ---------
        None : None
            This function does not take any parameter.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return any value.

        """

        # 0. Safely read in the BioGRID data
        datatypes = {
            '#BioGRID Interaction ID': int, 
            'Entrez Gene Interactor A': object,
            'Entrez Gene Interactor B': object,
            'BioGRID ID Interactor A': int,
            'BioGRID ID Interactor B': int,
            'Systematic Name Interactor A': object,
            'Systematic Name Interactor B': object,
            'Official Symbol Interactor A': object,
            'Official Symbol Interactor B': object,
            'Synonyms Interactor A': object,
            'Synonyms Interactor B': object,
            'Experimental System': object,
            'Experimental System Type': object,
            'Author': object,
            'Publication Source': object,
            'Organism ID Interactor A': int,
            'Organism ID Interactor B': int,
            'Throughput': object,
            'Score': object,
            'Modification': object,
            'Qualifications': object,
            'Tags': object,
            'Source Database': object,
            'SWISS-PROT Accessions Interactor A': object,
            'TREMBL Accessions Interactor A': object,
            'REFSEQ Accessions Interactor A': object,
            'SWISS-PROT Accessions Interactor B': object,
            'TREMBL Accessions Interactor B': object,
            'REFSEQ Accessions Interactor B': object,
            'Ontology Term IDs': object,
            'Ontology Term Names': object,
            'Ontology Term Categories': object,
            'Ontology Term Qualifier IDs': object,
            'Ontology Term Qualifier Names': object,
            'Ontology Term Types': object,
            'Organism Name Interactor A': object,
            'Organism Name Interactor B': object
        }

        print("reading in the BioGRID csv file...", end='')
        self.biog = pd.read_csv(self.config["projRootPath"] + self.config["bioGRID"], sep='\t', dtype=datatypes)
        print("done!")

        # 1. Ensure that only human interactomes are considered; 9606 is Human Organism ID
        self.biog = self.biog[(self.biog['Organism ID Interactor A'] == 9606) & (self.biog['Organism ID Interactor B'] == 9606)]

        # 2. Now we will build a graph that has genes/proteins as nodes and their interactions as edges
        self.interactomeGraph = nx.Graph()

        print("building the graph...", end='')
        #NOTE: The following might cause issue in later mapping
        for _, row in self.biog.iterrows():
            p1 = row['Official Symbol Interactor A'].replace('-', '_').replace('.', '_')
            p2 = row['Official Symbol Interactor B'].replace('-', '_').replace('.', '_')

            self.interactomeGraph.add_edge(p1, p2)
        print("done!")

        # 3. Remove self loops
        self.interactomeGraph.remove_edges_from(nx.selfloop_edges(self.interactomeGraph))

        # 4. Consider ONLY largest connected component
        print("looking for largest connected components...", end='')
        lcc = max(nx.connected_components(self.interactomeGraph), key=len)

        self.interactomeGraph = self.interactomeGraph.subgraph(lcc).copy()
        print("done!")

        # 5. Save graph to disk, for later use
        print("saving graph to disk...")
        nx.write_gml(self.interactomeGraph, self.config["projRootPath"] + self.config["graphsPath"]+'bioGRID_graph.gml')
        self.interactomeGraph = None
        print("Graph created and saved...")

    def prepareFeatures(self):
        """
        Assign the nodes of the PPI graph (from BioGRID) features based on the GDA (from DisGeNet).

        Parameter
        ---------
        None : None
            This function does not take any parameter.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return any value.

        """

        # Reading in the Interactome graph from disk
        print("reading in the PPI (BioGRID) graph...", end='')
        self.interactomeGraph = nx.read_gml(self.config["projRootPath"] + self.config["graphsPath"]+'bioGRID_graph.gml')
        print("done!")
        # Currently performing a temporary task, ideally I should compute this
        print("reading in the disease data...", end='')
        nf = pd.read_csv(self.config["projRootPath"] + "/data/NeDBIT_features/C0006142_Malignant_neoplasm_of_breast_features")
        print("done !")

        print("computing features...")
        degree = dict(zip(nf['name'], nf['degree']))
        ring = dict(zip(nf['name'], nf['ring']))
        NetRank = dict(zip(nf['name'], nf['NetRank']))
        NetShort = dict(zip(nf['name'], nf['NetShort']))
        HeatDiff = dict(zip(nf['name'], nf['HeatDiff']))
        InfoDiff = dict(zip(nf['name'], nf['InfoDiff']))

        del nf

        for node in self.interactomeGraph:
            self.interactomeGraph.nodes[node]['degree'] = degree[node]
            self.interactomeGraph.nodes[node]['ring'] = ring[node]
            self.interactomeGraph.nodes[node]['NetRank'] = NetRank[node]
            self.interactomeGraph.nodes[node]['NetShort'] = NetShort[node]
            self.interactomeGraph.nodes[node]['HeatDiff'] = HeatDiff[node]
            self.interactomeGraph.nodes[node]['InfoDiff'] = InfoDiff[node]

        degree      = []
        ring        = []
        NetRank     = []
        NetShort    = []
        HeatDiff    = []
        InfoDiff    = []

        for node in self.interactomeGraph:
            degree.append(self.interactomeGraph.nodes[node]['degree'])
            ring.append(self.interactomeGraph.nodes[node]['ring'])
            NetRank.append(self.interactomeGraph.nodes[node]['NetRank'])
            NetShort.append(self.interactomeGraph.nodes[node]['NetShort'])
            HeatDiff.append(self.interactomeGraph.nodes[node]['HeatDiff'])
            InfoDiff.append(self.interactomeGraph.nodes[node]['InfoDiff'])

        features = [degree, ring, NetRank, NetShort, HeatDiff, InfoDiff]
        print(
            "degree count {}\n".format(len(features[0])),
            "ring count {}\n".format(len(features[1])),
            "NetRank count {}\n".format(len(features[2])),
            "NetShort count {}\n".format(len(features[3])),
            "HeatDiff count {}\n".format(len(features[4])),
            "InfoDiff count {}".format(len(features[5]))
        )

        transformer = RobustScaler().fit(np.array(features))
        features = transformer.transform(np.array(features))

        i = 0
        for node in self.interactomeGraph:
            self.interactomeGraph.nodes[node]['degree'] = features[0][i]
            self.interactomeGraph.nodes[node]['ring'] = features[1][i]
            self.interactomeGraph.nodes[node]['NetRank'] = features[2][i]
            self.interactomeGraph.nodes[node]['NetShort'] = features[3][i]
            self.interactomeGraph.nodes[node]['HeatDiff'] = features[4][i]
            self.interactomeGraph.nodes[node]['InfoDiff'] = features[5][i]
            i += 1
        print("done !")

        # Adding the scores/output to the graph
        datatypes = {
            "name": object,
            "out": float
        }
        print("adding APU scores to the graph...", end='')
        scores = pd.read_csv(self.config["projRootPath"] + "/data/APU_scores/C0006142_Malignant_neoplasm_of_breast_features_Score.csv", dtype=datatypes)
        scores = dict(zip(scores['name'], scores['out']))

        for node in self.interactomeGraph:
            self.interactomeGraph.nodes[node]['score'] = scores[node]
        print("done !")

        print("saving graph to disk with features...", end='')
        # Save the graph to store a version with normalized features vectors
        nx.write_gml(self.interactomeGraph, self.config["projRootPath"] + self.config["graphsPath"]+'C0006142_nedbit.gml')
        # nx.write_gml(self.interactomeGraph, self.config["projRootPath"] + self.config["graphsPath"]+'bioGRID_graph.gml')
        print("done !")


def main():
    """
    This function acts as the starting point for the pre-processing of the raw data into tab-separated files of ready to consume data for machine learning.

    Parameter
    ---------
    None : None
        This function does not take any parameter.

    Raises
    ------
    None
        This function does not raise any error/exception.

    Returns
    -------
    None
        This function does not return any value.

    """

    while True:
        print("\nEnter 1 for generating the .tsv files (raw data processing)")
        print("Enter 2 for creating PPI (BioGRID) graph with node features")
        usrChc = int(input("Enter 0 to quit!:\t"))

        if usrChc == 1:
            print("Generating .tsv files from the DisGeNet SQLite database file!")
            cfgFile = str(Path(__file__).parent.absolute() / "configFiles/preproc_config.json")
            gdaObj = GDA(cfgFile)
            gdaObj.prepare_data()
            del cfgFile

        elif usrChc == 2:
            print("Generating the PPI graph with features!")
            cfgFile = str(Path(__file__).parent.absolute() / "configFiles/preproc_config.json")
            grphObj = DataGraph(str(cfgFile))
            grphObj.buildGraph()
            grphObj.prepareFeatures()
            del cfgFile

        elif usrChc == 0:
            print("Quitting the program...!")
            break


if __name__ == '__main__':
    main()
