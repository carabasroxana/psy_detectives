# psy-detectives
A. Reproducibility
A.1 Software and libraries
Platform: Google Colab (GPU enabled when available).
Core libraries: torch, transformers, datasets, scikit-learn, pandas, numpy, spacy.
spaCy: English pipeline used for sentence splitting (candidate segment extraction).
A.2 Data files and intermediate outputs
Inputs (task-provided): train_redacted.jsonl, dev_redacted.jsonl, test_redacted.jsonl.
Rehydrated outputs (starter pack): train_rehydrated.jsonl, dev_rehydrated.jsonl, test_rehydrated.jsonl.
Marker dictionary: psy_dict.csv.
Candidate segments export: candidate_segments_for_annotation.jsonl.
Binary baseline splits: train.csv, dev.csv (constructed from documents_for_annotation.jsonl + train_redacted.jsonl).
A.3 Rehydration procedure
Train: put train_redacted.jsonl inside the starter-pack folder as train_redacted.jsonl, then run:
python rehydrate_data.py → produces train_rehydrated.jsonl
DEV/TEST (exact Colab commands used):
git clone https://github.com/hide-ous/semeval26_starter_pack.git
cd semeval26_starter_pack


# Rehydrate DEV
cp /content/dev_redacted.jsonl train_redacted.jsonl
python rehydrate_data.py
cp train_rehydrated.jsonl /content/dev_rehydrated.jsonl


# Rehydrate TEST
cp /content/test_redacted.jsonl train_redacted.jsonl
python rehydrate_data.py
cp train_rehydrated.jsonl /content/test_rehydrated.jsonl

A.4 Randomness control
Train/dev split uses random_state=42 (stratified).
TinyLlama evaluation sample uses random_state=42.
No additional global seeding was applied beyond these explicit random_state settings.
B. Preprocessing details
B.1 Label normalization
Canonical label set: {Yes, No, Cant}
Mapping rule:
"Can't tell" → Cant
"Can’t tell" → Cant
"Cant" → Cant
B.2 Binary setting (BERT baseline)
Keep only labels in {Yes, No} and drop Cant.
Map: No → 0, Yes → 1.

C. Candidate segment extraction heuristics
Documents are sentence-split with spaCy. A sentence is selected if any trigger fires:
Modal trigger: lemma in
MODAL_LEMMAS = {must, should, could, might, may, would, will, can}
Question trigger: sentence ends with ?
Agent trigger: token in
GENERIC_AGENTS = {elite, elites, government, they, them, globalists, cabal}
For each selected sentence we store:
sent_index, segment, and flags: has_modal, is_question, has_agent.

D. Psycholinguistic marker dictionary (psy_dict.csv)
D.1 Tokenization and counting
Built from gold marker spans in train_rehydrated.jsonl.
Tokenization: re.findall(r"[A-Za-z']+", marker_text) then lowercase.
Update counts: token_cat_counts[token][marker_type] += 1.
D.2 Dominant category assignment
For each token:
total_count = sum_t count(token,t)
dominant_category = argmax_t count(token,t)
dominant_ratio = count(token,dominant_category) / total_count
D.3 Filtering thresholds
Keep token iff:
total_count >= 3
dominant_ratio >= 0.6
token not in this stopword-like set:
stop_like = {the, a, an, of, to, and, in, on, for, is, are, was, were, be, been, being, this, that, these, those, it, its, i, you, he, she, we, they, them, my, your, our, at, as, by, or, if, but, from, with, so, do, does, did, have, has, had, will, would, can, could, should, may, might, about, just, not, no, yes, also, there, their, than, then, when, what, which, who, how, why, because, into, out, up, down, over, under}

E. LIWC-style feature extraction
Using psy_dict.csv, compute per document (and optionally per candidate segment):
Category_count: number of matched tokens for that marker type
Category_ratio: Category_count / total_tokens
Categories: Actor, Action, Victim, Evidence, Threat, Effect

F. Baseline 1: Supervised BERT (binary Yes/No)
F.1 Data construction
Join documents_for_annotation.jsonl with train_redacted.jsonl on _id.
Filter to {Yes, No}.
Map: No→0, Yes→1.
Split:
train_test_split(test_size=0.2, stratify=label, random_state=42)
Train size: 3822, Dev size: 956
Binary label counts after filtering: No=2693, Yes=2085.
F.2 Tokenization and model
Model: bert-base-uncased
Tokenization: truncation=True, padding="max_length", max_length=128
Classifier: num_labels=2
F.3 Training hyperparameters
num_train_epochs=3
per_device_train_batch_size=16
per_device_eval_batch_size=32
learning_rate=2e-5
weight_decay=0.01
logging_steps=50
Evaluation: trainer.evaluate() after training.
F.4 Metrics
Accuracy
Weighted Precision / Recall / F1

G. Baseline 2: TinyLlama zero-shot (Yes/No/Cant)
G.1 Model and decoding
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Precision/device: torch.float16, device_map="auto"
Generation: greedy (do_sample=False), max_new_tokens=8
G.2 Prompt template
(Used exactly as in the notebook)
Instruction + label definitions + “Answer with only one word: Yes, No, or Cant.”
G.3 Output parsing rule
On generated text (lowercased):
if contains "yes" → Yes
else if contains "cant" or "can't" → Cant
else if contains "no" → No
else fallback → Cant
G.4 Evaluation protocol
Sample: N=500, random_state=42
Gold mapping includes "Can't tell" → Cant
Metrics: accuracy, classification report, confusion matrix

H. Additional results (non-essential but useful)
BERT dev performance: accuracy 0.7626, weighted F1 0.7605, weighted precision 0.7620, weighted recall 0.7626.
TinyLlama (N=500): accuracy 0.34; predicted distribution Yes=471, Cant=29, No=0.
TinyLlama confusion matrix (rows=gold, cols=pred; label order Yes/No/Cant):
[[165, 0, 13], [217, 0, 15], [89, 0, 1]]

G. TinyLlama prompting details (Another_copy_of_TinyLlama…)
prompt text (the instruction + label set)
decoding: greedy (do_sample=False), max_new_tokens=8
output parsing rules:
if contains “yes” → Yes
else if contains “no” → No
else if contains “cant/can’t” → Cant
else → Cant (fallback)
predicted distribution (Yes=471, Cant=29, No=0) on sample of 500.
