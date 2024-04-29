### Information about Data

* PPI of Human Genes only (_human interactome_)
* 19,761 genes; 6,78,932 undirected links (retaining only BioGRID genes)

[NIAPU - supplementary data](https://academic.oup.com/bioinformatics/article/39/2/btac848/7023926#supplementary-data)


| Sr.<br>No. | DisGeNet ID | Disease Name                   | Final<br>Gene<br>Count |
| :----: | :---------- | :----------------------------  | ---------------: |
| 1.     | C0006142    | Malignant Neoplasm of Breast   | 1025             |
| 2.     | C0036341    | Schizophrenia                  | 832              |
| 3.     | C0023893    | Liver cirrhosis                | 747              |
| 4.     | C0009402    | Colorectal carcinoma           | 672              |
| 5.     | C0376358    | Malignant Neoplasm of Prostate | 606              |
| 6.     | C0005586    | Bipolar disorder               | 451              |
| 7.     | C3714756    | Intellectual Disability        | 431              |
| 8.     | C0860207    | Drug induced liver disease     | 320              |
| 9.     | C0011581    | Depressive disorder            | 279              |
| 10.    | C0001973    | Chronic Alcoholic intoxication | 255              |


### Collecting the dataset

#### Part of the dataset (**PPI**) comes from BioGRID:

From the [all organism](https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.231/BIOGRID-ORGANISM-4.4.231.tab3.zip) file, we extract the `BIOGRID-ORGANISM-Homo_sapiens-4.4.231.tab3.txt` file.

#### The majority of the dataset (GDAs) are obtained from DisGeNet:

First, the SQLite file is downloaded from the [DisGeNet downloads](https://www.disgenet.org/downloads) page.

Then we use [an SQLite browser](https://sqlitebrowser.org/blog/portableapp-for-3-11-2-release-now-available/) to view the files from the above SQLite file.
After planning the SQL Queries based on the table schemas, we use the file:
`gene_disease/scripts/preprocess.py`


#### Decision: 18-Mar-2024
Due to inability to extract the correct dataset from the DisGeNet website (!),
We will continue with the file provided by the authors.

The coloumns in the various data files are as follows:

|`all_gene_disease_associations.tsv`|`disease_associations.tsv`|`gene_associations.tsv`|Columns I extracted|
|:----------------------------------|:-------------------------|:----------------------|:------------------|
|geneId                             |                          |geneId                 |geneID             |
|geneSymbol                         |                          |geneSymbol             |                   |
|                                   |                          |                       |geneName           |
|                                   |                          |                       |geneDescription    |
|DSI                                |                          |DSI                    |DSI                |
|DPI                                |                          |DPI                    |DPI                |
|diseaseId                          |diseaseId                 |                       |diseaseId          |
|diseaseName                        |diseaseName               |                       |diseaseName        |
|diseaseType                        |diseaseType               |                       |diseaseType        |
|diseaseClass                       |diseaseClass              |                       |diseaseClass       |
|diseaseSemanticType                |                          |                       |                   |
|                                   |                          |                       |diseaseClassName   |
|                                   |                          |                       |association        |
|                                   |                          |                       |associationType    |
|score                              |                          |                       |score              |
|EI                                 |                          |                       |EI                 |
|                                   |                          |                       |EL                 |
|                                   |                          |                       |year               |
|YearInitial                        |                          |                       |                   |
|YearFinal                          |                          |                       |                   |
|                                   |                          |                       |pmid               |
|NofPmids                           |NofPmids                  |NofPmids               |                   |
|NofSnps                            |                          |                       |                   |
|source                             |                          |                       |source             |
|                                   |NofGenes                  |                       |                   |
|                                   |diseaseSemanticType       |                       |                   |
|                                   |                          |PLI                    |                   |
|                                   |                          |protein_class_name     |                   |
|                                   |                          |protein_class          |                   |
|                                   |                          |NofDiseases            |                   |


#### The Data extraction SQL queries

The following creates a view for JOINs of `diseaseAttributes`, `disease2class`
and `diseaseClass`
```SQL
DROP VIEW DISEASE;
```
```SQL
CREATE VIEW DISEASE AS

SELECT diseaseAttributes.diseaseNID, diseaseAttributes.diseaseID,
diseaseAttributes.diseaseName, diseaseAttributes.type as diseaseType,
diseaseClass.diseaseClass, diseaseClass.diseaseClassName
FROM diseaseAttributes
JOIN disease2class ON diseaseAttributes.diseaseNID = disease2class.diseaseNID
JOIN diseaseClass ON disease2class.diseaseClassNID = diseaseClass.diseaseClassNID
```

Generating the data equivalent to the file `all_gene_disease_associations.tsv`:
```SQL
SELECT geneAttributes.geneID, geneAttributes.geneName, geneAttributes.geneDescription, 
geneAttributes.DSI, geneAttributes.DPI,
DISEASE.diseaseId, DISEASE.diseaseName, DISEASE.diseaseType, DISEASE.diseaseClass,
DISEASE.diseaseClassName,
geneDiseaseNetwork.source, geneDiseaseNetwork.association, 
geneDiseaseNetwork.associationType, geneDiseaseNetwork.pmid, geneDiseaseNetwork.score,
geneDiseaseNetwork.EL, geneDiseaseNetwork.EI, geneDiseaseNetwork.year
FROM geneAttributes
JOIN geneDiseaseNetwork
ON geneAttributes.geneNID = geneDiseaseNetwork.geneNID
JOIN DISEASE
ON geneDiseaseNetwork.diseaseNID = DISEASE.diseaseNID
```
The result of above needs to be combined with the files: `gene_associations.tsv`
and `disease_associations.tsv`.

#### Feature Extraction from raw data

The features used for the nodes (_genes_) in the interaction graph, are:

1. [Heat Diffusion](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5638226/pdf/pcbi.1005598.pdf)
2. Balanced Heat Diffusion
3. [NetShort](http://www.datalab.uci.edu/papers/white_smyth.pdf)
4. [NetRing](https://core.ac.uk/reader/16409665)

How do we calculate each?
