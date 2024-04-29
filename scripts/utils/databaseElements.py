import sqlite3
import pandas as pd
import json
from pathlib import Path


class GeneDiseaseAssociation:
    """
    This class performs all the raw data extraction from the DisGeNet database
    and prepares the Gene-Disease Association file.
    err checking. Check if all the requisite files are available
    1. Create the Database connections/objects
    2. Fire the View creation and data extraction queries
    3. Generate the Pandas dataframe and save as a tsv file

    Attributes
    ----------
    None : None
        The GeneDiseaseAssociation class does not have any class variable.

    Methods
    -------
    __init__(self):
        Load the pre-process config json file and construct object for the SQLiteHandler class.
    prepare_data(self):
        Connect to the SQLite database file, test and create the DISEASE view, and finally create the GDA tsv file by querying joins on the the database.

    """
    def __init__(self, configFile):
        """
        Open the json file pertaining to the pre-processing configuration. Further, create an object for the SQLHandler class.

        Parameter
        ---------
        configFile : str
            The absolute path to the pre-processing .json configuration file.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return anything.

        """

        super(GeneDiseaseAssociation, self).__init__()
        with open(configFile, "r") as cfg:
            self.config = json.load(cfg)
        self.dbObj = SQLiteHandler()
        # gene = pd.read_csv("./data/gene_associations.tsv", sep="\t")
        
    def prepare_data(self):
        """
        Establish a connection with the database file; create the DISEASE view on the database and then create a join to extract the Gene-Disease Association information. At the end save the data as tab separated file.

        Parameter
        ---------
        None : None
            This function has no parameter.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return anything.

        """

        self.dbObj.connectionManager(self.config["projRootPath"] + self.config["dbFileName"])
        exist = self.dbObj.sqlQuery(self.config["existViewDiseaseQ"])
        if len(exist) == 1:
            self.dbObj.sqlQuery(self.config["dropViewDiseaseQ"])
        self.dbObj.sqlQuery(self.config["makeViewDiseaseQ"])
        results = self.dbObj.sqlQuery(self.config["geneDiseaseEquivQ"])

        cols = [
            "geneID", "geneName", "geneDescription", "DSI", "DPI", "diseaseId",
            "diseaseName", "diseaseType", "diseaseClass", "diseaseClassName",
            "source", "association", "associationType", "pmid", "score", "EL", "EI",
            "year"
        ]
        resultsDF = pd.DataFrame(results, columns=cols)
        outFilePath = Path(self.config["projRootPath"] + self.config["gdaTSV"])

        print("writing results to csv file...")
        
        resultsDF.to_csv(str(outFilePath), sep='\t', columns=cols, mode="w")
        del self.dbObj


class SQLiteHandler:
    """
    This class is meant to handle connections with an SQLite database file.
    Tasks such establishing/closing connections, firing queries, saving results
    as .tsv files are covered.

    Attributes
    ----------
    None : None
        The SQLiteHandler class does not have any class variable.

    Methods
    -------
    __init__(self):
        The object constructor for the SQLiteHandler class. It creates instance variables for connection and cursor.
    __del__(self):
        Close the cursor, connection and free the variables.
    connectionManager(self, dbFileName):
        Connect to the sqlite database file and create a cursor.
    sqlQuery(self, qString):
        Fire the actual SQL query on the SQLite database.

    """

    def __init__(self):
        """
        This is the constructor for the SQLiteHandler class; declares the instance variables for the connection and cursor.

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

        super(SQLiteHandler, self).__init__()
        self.conn = None
        self.cur = None
        #print("executed __init__")

    def __del__(self):
        """
        This function makes sure that the cursor and connection to the database file are closed safely.

        Parameter
        ---------
        None : None
            The function does not have any parameter other than a refernce to self.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This funciton does not return any value.

        """

        print("closing cursor...")
        self.cur.close()
        print("closing connection to database...")
        self.conn.close()
        self.cur = None
        self.conn = None
        #print("executed __del__")

    def connectionManager(self, dbFileName):
        """
        Connect to an sqlite database file and create a cursor to it.

        Parameter
        ---------
        dbFileName : str
            This string provides the absolute path to the sqlite database file.

        Raises
        ------
        None
            This function does not raise any error/exception.

        Returns
        -------
        None
            This function does not return any value.

        """

        self.conn = sqlite3.connect(dbFileName)
        self.cur = self.conn.cursor()
        print("established connection and created cursor...")

    def sqlQuery(self, qString):
        """
        This function executes an SQL query on an SQLite connection's cursor and return the result as a list of tuples.

        Parameter
        ---------
        qString : str
            The SQL query to be fired.

        Raises
        ------
        None
            This function does not throw any error/exception.

        Returns
        -------
        list of tuples
            The result of the query is returned as a list of tuples.

        """

        #print("executed sqlQ")
        #print(qString)
        self.cur.execute(qString)
        print("executing SQL query...")
        #return self.cur.fetchmany(10)
        return self.cur.fetchall()
