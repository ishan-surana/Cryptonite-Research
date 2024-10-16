# Ontology-Based Malware Classification System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Background](#background)
3. [Key Features](#key-features)
4. [Directory Structure](#directory-structure)
5. [Usage](#usage)
6. [Ontology Design](#ontology-design)
7. [Dynamic Rule Generation](#dynamic-rule-generation)
8. [Validation and Testing](#validation-and-testing)
9. [Results](#results)
10. [Conclusions](#conclusions)
11. [License](#license)

## Project Overview
This project implements a dynamic ontology-based system for classifying executable files as either benign or malware. The focus is on creating a robust ontology that not only captures essential features of executable files but also incorporates dynamic rule-based classification derived from decision trees. It also addresses the problem of limited reasoning capability and lack of understanding by
incorporating semantic feature and inference relations like inverses for bi-directional querying and clearer insights, as well as the problem of complexity by reducing classification depth. The methodology groups various features into classes to ensure visible relations, and derives hidden relations using correlation factors between features of different classes, implementing them along side the derived rules. It also reduces load by reducing depth and increasing feature classifications.

The system aims to enhance traditional ontology approaches by integrating dynamic reasoning capabilities, detailed semantic relations between features, deriving and incorporating hidden realtionships, and adding influence relations between data points and features, allowing for more adaptable and scalable malware classification.

## Background
In the cybersecurity domain, effective malware detection is crucial for protecting systems from malicious threats. Traditional approaches often rely on static definitions and manual rule creation, which can be inadequate in adapting to evolving threats. They are also quite unintuitive and hard to understand owing to a lack of semantic relationships, and can also be quite resource intensive to work with. This project addresses these challenges by leveraging ontologies and machine learning techniques to create a more dynamic classification system.

## Key Features
- **Dynamic Ontology**: Incorporates features, relationships, and classifications of executable files, allowing for real-time updates based on new data.
- **Flexible Reasoning**: Implements dynamic rule updates and semantic relationships to improve classification accuracy and adaptability.
- **Correlation Analysis**: Identifies hidden relationships between features based on correlation metrics, enhancing the interpretability of the ontology.
- **Semantic and Extended Relations**: Establishes detailed semantic and extended relationships between features, allowing for improved inference and better understanding of feature interactions. This enhances classification decisions and insights into malware behavior.
- **Visible and Hidden Relations**: Organizes and displays clear relationships between features of same class while also deriving hidden relationships using correlation factors, enhancing the ontology's interpretability and depth.

## Directory Structure
- **`experimentation`**: Files from the inception stage of the project, contains 2 directories - `owl files` and `outputs`, having various files created using sample and actual datasets to visualise the effects of hierarchy, properties and rules in ontologies, using [Protege](https://protege.stanford.edu/software.php#desktop-protege) and [WebProtege](https://webprotege.stanford.edu).
- **`references`**: The collection of referred research papers for the project, in PDF format.
- **`data`**: The compilation of all datasets collected for the topic's purpose. The links to the main datsets are [data, data_modified](https://raw.githubusercontent.com/Kiinitix/Malware-Detection-using-Machine-learning/refs/heads/main/Dataset/data.csv), 
[ClaMP dataset](https://raw.githubusercontent.com/urwithajit9/ClaMP/refs/heads/master/dataset/ClaMP_Integrated-5184.csv), [sample_analysis_data, mal-api-2019](https://github.com/ocatak/malware_api_class), [df_m](https://raw.githubusercontent.com/HudaLughbi/CybAttT/refs/heads/main/df_m.csv)
- **`code and work`**: The main project folder, contains the following:-
    + [progress notebook](code%20and%20work/Malware_analysis_KG+KR.ipynb)
    + [final script](code%20and%20work/final_script.py)
    + [dataset used](code%20and%20work/datasets/ClaMP_Integrated-5184.csv)
    + [Detailed report of improvements](code%20and%20work/Improvements%20Report.docx)
    + **Ontology OWLs and output files**:
        - [Ontology having hidden relations](code%20and%20work/outputs/ontology_with_hidden.owl)
        - [Ontology without hidden relations](code%20and%20work/outputs/ontology_wo_hidden.owl)
        - Decision tree visualisation, [PDF](code%20and%20work/outputs/decision_tree_graph.pdf) and [PNG](code%20and%20work/outputs/decision_tree_graph.png)
    + **OntoMetric reports**:
        - Ontology having hidden relations [PDF](code%20and%20work/OntoMetrics%20reports/Ontology%20Metrics%20(with%20hidden).pdf) and [XML](code%20and%20work/OntoMetrics%20reports/Ontology%20Metrics%20(with%20hidden).xml)
        - Ontology without hidden relations [PDF](code%20and%20work/OntoMetrics%20reports/Ontology%20Metrics%20(without%20hidden).pdf) and [XML](code%20and%20work/OntoMetrics%20reports/Ontology%20Metrics%20(without%20hidden).xml)

## Usage
1. Clone the repository to your local machine.
2. Install all the required Python libraries and modules.
3. The notebook is a progress record and not compulsory to run. Each section is titled with the stage in work progress. Execute the code cell by cell, and follow any instructions provided in the comments (for eg, regarding setup and usage of Graphviz). 
3. Run the provided Python script. The script will:
   - Load and preprocess the dataset.
   - Train a decision tree classifier on the data.
   - Extract classification rules from the trained model.
   - Create and populate an ontology to represent the executable files and their features, implementing classes, relations and properties.
   - Integrate dynamic rule generation based on the decision tree, as well as hidden inferences based on feature correlations.
   - Validate the ontology using the Pellet reasoner.
   - Save the ontology in OWL format.

## Ontology Design
The ontology defines various classes representing executable files, header fields, sections, packer information, file size, and malware classifications. Relationships between these classes are also established, allowing for a structured representation of the data. Properties are defined for each class, and visible rules are inferred by the decision tree and added to ontology in SWRL format. Hidden inferences between the features are obtained using correlation values and appended to the ontology. 

### Key Classes
- **`ExecutableFile`**: Represents individual executable files.
- **`HeaderFields`**: Captures metadata related to the file's header.
- **`Sections`**: Contains information about the different sections within an executable file.
- **`PackerInformation`**: Details about the packer used for the executable.
- **`FileSize`**: Represents size-related attributes of the executable.
- **`MalwareClassification`**: Differentiates between `Benign` and `Malware` classifications.

## Dynamic Rule Generation
The project implements a function to extract rules from the decision tree classifier. These rules are then converted into SWRL (Semantic Web Rule Language) format to be integrated into the ontology. The dynamic nature of the rule generation allows for real-time updates based on the current state of the data.

## Validation and Testing
After the ontology is constructed and populated with instances from the dataset, the Pellet reasoner is used to validate the ontology. The reasoner checks for consistency and derives any implicit relationships or classifications based on the defined rules and axioms.

### Example Validation
As an example, the following check is performed to validate the classification of files:
```python
for i in range(len(df)):
    exec(f"classification = str(onto.File_{i}.isClassifiedAs[0]).split('_')[1].split('.')[1]")
    if classification != df['class'][i]:
        print(f"File {i} misclassified as {classification}.")
```

## Results
The accuracy of the decision tree classifier is about `97.8%`. The output provides insights into the effectiveness of the classification model, and any misclassifications are also flagged for further analysis. The decision tree is available in both PDF and PNG formats for visualisation. Validated ontology of both types are present - with and without considering hidden relations.
A detailed report of the improvements is specified in the document [here](code%20and%20work/Improvements%20Report.docx). The performance of the ontology is graded based on three key benchmarks:-
1) **Discovery of hidden relationships & extra knowledge gathered**: Hidden correlations are derived based on feature correlations and appended to ontology. The effect is visible in the results.
2) **Improvment in results with the new methodology**: Compared to various ontologies for reference, our methodology surpasses all of them in almost every attribute measured, using the standard tool [OntoMetrics](https://ontometrics.informatik.uni-rostock.de/ontologymetrics/). The results are as follows:-

    | Metric | Chowdhury & Bhowmik | MALOnt | Swimmer’s Ontology | *Our Ontology (w/o hidden correlations)* | *Our Ontology (with hidden    correlations)* |
    | --- | --- | --- | --- | --- | --- |
    | Attribute richness | 0.206897 | 0.191176 | 0.0 | **7.555556** | **7.555556** |
    | Class richness | 0.310345 | **0.970588** | 0.0 | 0.777778 | 0.777778 |
    | Inheritance richness | 0.965517 | 0.676471 | 0.941176 | 0.888889 | **1.111111** |
    | Relationship richness | 0.333333 | 0.402597 | 0.0 | **0.6** | 0.545455 |
    | Average population | 4.896552 | 3.897059 | 0.0 | **3473.555556** | **3473.555556** |
    
3) **Ontology verification and validation**: This was achieved using the [Pellet](https://github.com/stardog-union/pellet) reasoner. The successful run and no adjustments to the ontology components signified consistency, completeness and correctness of the ontology.

## Conclusions
The dynamic ontology-based system developed for classifying executable files significantly enhances malware detection. Key outcomes include:
- **Improved Interpretability**: Clear visibility of both visible and hidden relationships enhances understanding of feature interactions.
- **Dynamic Rule Generation**: Enables real-time updates to classification rules, allowing for adaptability to evolving threats.
- **Robust Ontology Design**: Well-defined classes and properties support effective data representation and relationship capturing.
- **Benchmark Performance**: Outperformed existing ontologies in key metrics, validating its effectiveness.
- **Validation Success**: Pellet reasoner confirmed the ontology's consistency and correctness.
This project marks a significant advancement in adaptable and scalable malware classification, paving the way for future enhancements and broader applications.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
