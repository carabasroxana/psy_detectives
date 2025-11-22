# PsyCoMark Task 10 – Data Processing & Annotation Pipeline  
Author: Roxana Carabaș

This repository contains all the data preparation, preprocessing, feature extraction, and annotation export work completed for **SemEval Task 10: PsyCoMark – Psycholinguistic Conspiracy Marker Extraction and Detection**.

The main focus of this repository is on:

- Preprocessing the PsyCoMark dataset  
- Rehydrating the redacted Reddit data  
- Extracting candidate segments using heuristic rules  
- Generating a psycholinguistic dictionary from annotated spans  
- Computing document- and segment-level LIWC‐style features  
- Exporting documents and segments for manual annotation  
- Organizing all outputs clearly for inspection and grading  

---

## Repository Structure

psy_detectives/
│
├── data/
│ ├── train_redacted.jsonl
│ ├── train_rehydrated.jsonl
│ ├── candidate_segments_for_annotation.jsonl
│ ├── documents_for_annotation.jsonl
│ ├── psy_dict.csv
│
├── notebooks/
│ └── psycomark_data_pipeline.ipynb
│
├── README.md
└── (other project files)


### Why both **data/** and **notebooks/** folders?

- **data/** contains all input and output data files used or produced by the pipeline  
  (redacted dataset, rehydrated dataset, psycholinguistic dictionary, annotation exports)
- **notebooks/** contains the complete Colab notebook with the full pipeline  

---

## How to Download / Rehydrate the Data

### 1. Download the redacted training data  
This file is already included in the repository: data/train_redacted.jsonl
It contains the labels and marker spans, but **no text** (text must be rehydrated).

### 2. Rehydrate the data  
Open the notebook: notebooks/psycomark_data_pipeline.ipynb
Run the cell that calls the official rehydration script: this will generate: data/train_rehydrated.jsonl which includes full Reddit submission text.

### 3. Run preprocessing, feature extraction & exports  
The notebook performs:
- Text cleaning  
- Sentence splitting (spaCy)  
- Heuristic candidate sentence extraction  
- Psycholinguistic dictionary construction  
- LIWC-style feature extraction  
- Exporting data for annotation tools  

All final data files are saved automatically into the `data/` folder.

---

## What This Project Produces

### ✔ `candidate_segments_for_annotation.jsonl`
Sentences likely to contain conspiracy-related markers.

### ✔ `documents_for_annotation.jsonl`
Full documents for manual labeling.

### ✔ `psy_dict.csv`
A custom LIWC-style dictionary automatically built from the dataset’s marker spans.

### ✔ Document-level & segment-level features  
Generated inside the notebook.

---

## How to Reproduce Everything

1. Open the notebook in Colab  
2. Run all cells in order  
3. All outputs will appear in the `data/` directory  
4. You can inspect or download all generated data files

---

## Notes

- The public SemEval training split contains **one annotation per document**, so inter-annotator agreement (IAA) is not computable on this subset.  
- All scripts and exports in this repository are reproducible by running the notebook.





