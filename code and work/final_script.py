import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
from owlready2 import *

# Load the dataset
df = pd.read_csv("datasets/ClaMP_Integrated-5184.csv")
df['packer_type'] = df['packer_type'].astype('category').cat.codes
df['packer_type'] = df['packer_type'].apply(lambda x: int(x) if str(x).isdigit() else 0)  # Replace strings with default int value
df['packer'] = df['packer'].apply(lambda x: int(x) if str(x).isdigit() else 0)
df['packer_type'] = df['packer_type'].astype(int)
df['packer'] = df['packer'].astype(int)

# Preprocessing: Remove non-numeric columns and handle missing values
df = df.fillna(df.median())

# Split dataset into features (X) and target (y)
X = df.drop(columns=["class"])
y = df["class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = clf.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy}")

# Extract rules from the trained decision tree
def tree_to_rules(tree, feature_names):
    """ Recursively convert a decision tree into a list of rules. """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules_list = []

    def recurse(node, depth, current_rule):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = current_rule + [f"{name} <= {threshold}"]
            right_rule = current_rule + [f"{name} > {threshold}"]

            recurse(tree_.children_left[node], depth + 1, left_rule)
            recurse(tree_.children_right[node], depth + 1, right_rule)
        else:
            class_label = tree_.value[node].argmax()
            rule = " AND ".join(current_rule) + f" THEN class = {class_label}"
            rules_list.append(rule)

    recurse(0, 1, [])
    return rules_list

rules = tree_to_rules(clf, X.columns)

# Convert rules to SWRL format for OWL ontology
def convert_conditions_to_swr(rule):
    """Convert Python-style rule conditions to SWRL rule format."""
    swrl_conditions = []

    # Split conditions by 'and' (conditions must be combined with AND)
    conditions, classification = rule.split("THEN")
    conditions = conditions.strip().split("AND")

    for condition in conditions:
        # Split each condition into a feature, operator, and value
        feature, operator, value = condition.split()
        value = value.strip()  # Remove extra spaces

        # SWRL supports only some operators, so we map Python operators to SWRL compatible ones
        if operator == "<=":
            swrl_conditions.append(f"lessThanOrEqual({feature}, {value})")
        elif operator == ">":
            # Depending on the ontology, you may need to model 'greaterThan' in SWRL
            swrl_conditions.append(f"greaterThan({feature}, {value})")

    # Join all conditions with SWRL's conjunction operator (^)
    swrl_condition_string = " ^ ".join(swrl_conditions)

    # Extract the class (0 for Benign, 1 for Malware)
    classification = classification.strip().split(" = ")[-1]
    if classification == "1":
        swrl_conclusion = "isClassifiedAs(?x, Malware)"
    else:
        swrl_conclusion = "isClassifiedAs(?x, Benign)"

    return f"{swrl_condition_string} -> {swrl_conclusion}"

# Preprocess the data for ontology development
column_groups = {
    "Header Fields": ["e_cblp", "e_cp", "e_cparhdr", "e_maxalloc", "e_sp", "e_lfanew", "NumberOfSections", "CreationYear"] + [f"FH_char{i}" for i in range(15)],
    "Sections": ["sus_sections", "non_sus_sections", "MajorLinkerVersion", "MinorLinkerVersion", "SizeOfCode", "SizeOfInitializedData", "SizeOfUninitializedData", "AddressOfEntryPoint", "BaseOfCode", "BaseOfData", "ImageBase", "SectionAlignment", "FileAlignment", "MajorOperatingSystemVersion", "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion", "MajorSubsystemVersion", "MinorSubsystemVersion", "SizeOfImage", "SizeOfHeaders", "CheckSum", "Subsystem"] + [f"OH_DLLchar{i}" for i in range(11)],
    "Packer Information": ["packer_type", "packer"],
    "File Size": ["filesize", "SizeOfStackReserve", "SizeOfStackCommit", "SizeOfHeapReserve", "SizeOfHeapCommit"],
    "Text/Data Information": ["E_text", "E_data", "E_file"],
    "File Info": ["fileinfo"]
}

corr_matrix = df.corr()
# Hidden relationships based on data patterns, i.e., high correlation between features not in the same group
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        #  if the correlation > 0.5 and the pair is not the same feature or in same group as per the dictionary
        if corr_matrix.iloc[i, j] > 0.5 and [key for key, value in column_groups.items() if corr_matrix.columns[i] in value] != [key for key, value in column_groups.items() if corr_matrix.columns[j] in value] and 'class' not in [corr_matrix.columns[i], corr_matrix.columns[j]]:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("Highly Correlated Features:")
for pair in high_corr_pairs:
    print(f"{pair[0]} - {pair[1]}: {pair[2]}")

# Remove existing onolotgy
# print(list(default_world.ontologies))
onto = default_world.get_ontology("executable_ontology.owl")
onto.destroy()

# Load or create a new ontology
onto = get_ontology("executable_ontology.owl")

# Define ontology structure
with onto:
    class ExecutableFile(Thing):
        pass

    class HeaderFields(Thing):
        pass

    class Sections(Thing):
        pass

    class PackerInformation(Thing):
        pass

    class FileSize(Thing):
        pass

    class MalwareClassification(Thing):
        pass

    class Benign(MalwareClassification):
        pass

    class Malware(MalwareClassification):
        pass

# Define Object Properties (Relationships)
with onto:
    # Relationship to define header fields, sections, etc.
    class hasHeaderField(ExecutableFile >> HeaderFields):
        pass

    class isHeaderFieldOf(HeaderFields >> ExecutableFile):
        inverse_property = hasHeaderField  # Refer to the property object, not a string

    class hasSection(ExecutableFile >> Sections):
        pass

    class isSectionOf(Sections >> ExecutableFile):
        inverse_property = hasSection  # Refer to the property object, not a string

    class isClassifiedAs(ExecutableFile >> MalwareClassification):
        pass

    class classifiesFile(MalwareClassification >> ExecutableFile):
        inverse_property = isClassifiedAs  # Refer to the property object

    class hasPackerInfo(ExecutableFile >> PackerInformation):
        pass

    class isPackerInfoOf(PackerInformation >> ExecutableFile):
        inverse_property = hasPackerInfo  # Refer to the property object

    class hasFileSize(ExecutableFile >> FileSize):
        pass

    class isFileSizeOf(FileSize >> ExecutableFile):
        inverse_property = hasFileSize  # Refer to the property object

    # Relations for semantic meaning and influences
    class influencesField(ObjectProperty):
        domain = [HeaderFields, Sections, PackerInformation]
        range = [MalwareClassification, FileSize]
        pass

    class correlatesWith(ObjectProperty):
        domain = [HeaderFields, Sections, PackerInformation]
        range = [MalwareClassification, FileSize]
        pass

    # Hidden properties for correlation
    class hiddenCorrelationWith(AnnotationProperty):
        pass

# Define Data Properties (Attributes)
with onto:
    # Header Fields
    class e_cblp(HeaderFields >> int, FunctionalProperty):
        pass

    class e_cp(HeaderFields >> int, FunctionalProperty):
        pass

    class e_cparhdr(HeaderFields >> int, FunctionalProperty):
        pass

    class e_maxalloc(HeaderFields >> int, FunctionalProperty):
        pass

    class e_sp(HeaderFields >> int, FunctionalProperty):
        pass

    class e_lfanew(HeaderFields >> int, FunctionalProperty):
        pass

    class NumberOfSections(HeaderFields >> int, FunctionalProperty):
        pass

    class CreationYear(HeaderFields >> int, FunctionalProperty):
        pass

    # File Header Characteristics (e.g., FH_char0 to FH_char14)
    for i in range(15):
        exec(f'class FH_char{i}(HeaderFields >> int, FunctionalProperty): pass')

    # Section Information
    class MajorLinkerVersion(Sections >> int, FunctionalProperty):
        pass

    class MinorLinkerVersion(Sections >> int, FunctionalProperty):
        pass

    class SizeOfCode(Sections >> int, FunctionalProperty):
        pass

    class SizeOfInitializedData(Sections >> int, FunctionalProperty):
        pass

    class SizeOfUninitializedData(Sections >> int, FunctionalProperty):
        pass

    class AddressOfEntryPoint(Sections >> int, FunctionalProperty):
        pass

    class BaseOfCode(Sections >> int, FunctionalProperty):
        pass

    class BaseOfData(Sections >> int, FunctionalProperty):
        pass

    class ImageBase(Sections >> int, FunctionalProperty):
        pass

    class SectionAlignment(Sections >> int, FunctionalProperty):
        pass

    class FileAlignment(Sections >> int, FunctionalProperty):
        pass

    class MajorOperatingSystemVersion(Sections >> int, FunctionalProperty):
        pass

    class MinorOperatingSystemVersion(Sections >> int, FunctionalProperty):
        pass

    class MajorImageVersion(Sections >> int, FunctionalProperty):
        pass

    class MinorImageVersion(Sections >> int, FunctionalProperty):
        pass

    class MajorSubsystemVersion(Sections >> int, FunctionalProperty):
        pass

    class MinorSubsystemVersion(Sections >> int, FunctionalProperty):
        pass

    class SizeOfImage(Sections >> int, FunctionalProperty):
        pass

    class SizeOfHeaders(Sections >> int, FunctionalProperty):
        pass

    class CheckSum(Sections >> int, FunctionalProperty):
        pass

    class Subsystem(Sections >> int, FunctionalProperty):
        pass

    # DLL Characteristics (e.g., OH_DLLchar0 to OH_DLLchar10)
    for i in range(11):
        exec(f'class OH_DLLchar{i}(Sections >> int, FunctionalProperty): pass')

    # Packer Information
    class packer_type(PackerInformation >> int, FunctionalProperty):
        pass

    class packer(PackerInformation >> int, FunctionalProperty):
        pass

    # File Size and Related Properties
    class SizeOfStackReserve(FileSize >> int, FunctionalProperty):
        pass

    class SizeOfStackCommit(FileSize >> int, FunctionalProperty):
        pass

    class SizeOfHeapReserve(FileSize >> int, FunctionalProperty):
        pass

    class SizeOfHeapCommit(FileSize >> int, FunctionalProperty):
        pass

    class filesize(FileSize >> float, FunctionalProperty):
        pass

    # Suspicious Sections
    class sus_sections(Sections >> int, FunctionalProperty):
        pass

    class non_sus_sections(Sections >> int, FunctionalProperty):
        pass

    # Text/Data Information
    class E_text(ExecutableFile >> float, FunctionalProperty):
        pass

    class E_data(ExecutableFile >> float, FunctionalProperty):
        pass

    class E_file(ExecutableFile >> float, FunctionalProperty):
        pass

    # File Info
    class fileinfo(ExecutableFile >> int, FunctionalProperty):
        pass

# Dynamic Rule Integration into Ontology
with onto:
    for rule in rules:
        swrl_rule = convert_conditions_to_swr(rule)
        try:
            dynamic_rule = Imp().set_as_rule(swrl_rule)  # Add SWRL-style rule
        except Exception as e:
            print(rule)
            print(swrl_rule)
            print(f"Error adding rule: {e}")

# Add hidden relationships based on data patterns
with onto:
    for pair in high_corr_pairs:
        feature_1, feature_2 = getattr(onto, pair[0]), getattr(onto, pair[1])
        #  create a new relationship between the features
        feature_1.hiddenCorrelationWith.append(feature_2)

# replace class column of df from 0/1 to Benign/Malware
df['class'] = df['class'].replace({0: 'Benign', 1: 'Malware'})

# Populate the ontology using data from the DataFrame
for index, row in df.iterrows():
    with onto:
        # Create an instance of ExecutableFile
        exec_file = ExecutableFile(f"File_{index}")

        # Set relevant attributes (header fields, sections, packer info, etc.)
        header_fields = HeaderFields(f"Header_{index}")
        exec_file.hasHeaderField = [header_fields]

        header_fields.e_cblp = int(row['e_cblp'])
        header_fields.e_cp = int(row['e_cp'])
        header_fields.e_cparhdr = int(row['e_cparhdr'])
        header_fields.e_maxalloc = int(row['e_maxalloc'])
        header_fields.e_sp = int(row['e_sp'])
        header_fields.e_lfanew = int(row['e_lfanew'])
        header_fields.NumberOfSections = int(row['NumberOfSections'])
        header_fields.CreationYear = int(row['CreationYear'])

        # File Header Characteristics (e.g., FH_char0 to FH_char14)
        for i in range(15):
            header_fields.__setattr__(f"FH_char{i}", int(row[f"FH_char{i}"]))

        # Create Sections instance and assign values
        sections = Sections(f"Section_{index}")
        exec_file.hasSection = [sections]

        sections.sus_sections = int(row['sus_sections'])
        sections.non_sus_sections = int(row['non_sus_sections'])

        sections.MajorLinkerVersion = int(row['MajorLinkerVersion'])
        sections.MinorLinkerVersion = int(row['MinorLinkerVersion'])
        sections.SizeOfCode = int(row['SizeOfCode'])
        sections.SizeOfInitializedData = int(row['SizeOfInitializedData'])
        sections.SizeOfUninitializedData = int(row['SizeOfUninitializedData'])
        sections.AddressOfEntryPoint = int(row['AddressOfEntryPoint'])
        sections.BaseOfCode = int(row['BaseOfCode'])
        sections.BaseOfData = int(row['BaseOfData'])
        sections.ImageBase = int(row['ImageBase'])
        sections.SectionAlignment = int(row['SectionAlignment'])
        sections.FileAlignment = int(row['FileAlignment'])
        sections.MajorOperatingSystemVersion = int(row['MajorOperatingSystemVersion'])
        sections.MinorOperatingSystemVersion = int(row['MinorOperatingSystemVersion'])
        sections.MajorImageVersion = int(row['MajorImageVersion'])
        sections.MinorImageVersion = int(row['MinorImageVersion'])
        sections.MajorSubsystemVersion = int(row['MajorSubsystemVersion'])
        sections.MinorSubsystemVersion = int(row['MinorSubsystemVersion'])
        sections.SizeOfImage = int(row['SizeOfImage'])
        sections.SizeOfHeaders = int(row['SizeOfHeaders'])
        sections.CheckSum = int(row['CheckSum'])
        sections.Subsystem = int(row['Subsystem'])

        # DLL Characteristics (e.g., OH_DLLchar0 to OH_DLLchar10)
        for i in range(11):
            sections.__setattr__(f"OH_DLLchar{i}", int(row[f"OH_DLLchar{i}"]))

        # Create PackerInformation instance and assign values
        packer_info = PackerInformation(f"Packer_{index}")
        exec_file.hasPackerInfo = [packer_info]
        packer_info.packer_type = int(row['packer_type'])
        packer_info.packer = int(row['packer'])

        # Create FileSize instance and assign values
        file_size = FileSize(f"FileSize_{index}")
        exec_file.hasFileSize = [file_size]
        file_size.filesize = float(row['filesize'])
        file_size.SizeOfStackReserve = int(row['SizeOfStackReserve'])
        file_size.SizeOfStackCommit = int(row['SizeOfStackCommit'])
        file_size.SizeOfHeapReserve = int(row['SizeOfHeapReserve'])
        file_size.SizeOfHeapCommit = int(row['SizeOfHeapCommit'])

        # Text/Data Information
        exec_file.E_text = float(row['E_text'])
        exec_file.E_data = float(row['E_data'])
        exec_file.E_file = float(row['E_file'])

        # File Info
        exec_file.fileinfo = int(row['fileinfo'])

        # Create MalwareClassification instance and assign values
        if row['class'] == 'Malware':
            classification = Malware(f"Malware_{index}")
        else:
            classification = Benign(f"Benign_{index}")

        exec_file.isClassifiedAs = [classification]

# Run the reasoner to validate the ontology
with onto:
    sync_reasoner_pellet(debug=1)

# Check for changes in classification by reasoner
for i in range(len(df)):
    exec(f"classification = str(onto.File_{i}.isClassifiedAs[0]).split('_')[1].split('.')[1]")
    if classification != df['class'][i]:
        print(f"File {i} misclassified as {classification}.")

# Save the ontology with new rules
onto.save(file="executable_ontology_full_dynamic.owl", format="rdfxml")
