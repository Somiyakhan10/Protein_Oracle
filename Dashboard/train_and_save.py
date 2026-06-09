"""
train_and_save.py
─────────────────
Run this ONCE to:
  1. Fetch 198 reviewed human enzyme sequences from UniProt REST API
  2. Generate ESM-2 embeddings (facebook/esm2_t6_8M_UR50D)
  3. Train a Random Forest classifier
  4. Save  rf_model.joblib  and  scaler.joblib  for the Streamlit dashboard

Usage:
    python train_and_save.py
"""

import requests, time, joblib
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ─── 1. FETCH FROM UNIPROT ────────────────────────────────────────────────────
QUERIES = {
    'Kinase':        'kinase AND reviewed:true AND organism_id:9606',
    'Phosphatase':   'phosphatase AND reviewed:true AND organism_id:9606',
    'Protease':      'protease AND reviewed:true AND organism_id:9606',
    'Dehydrogenase': 'dehydrogenase AND reviewed:true AND organism_id:9606',
    'Transferase':   'transferase AND reviewed:true AND organism_id:9606',
}

def fetch_uniprot(query: str, size: int = 40):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {'query': query, 'format': 'json', 'size': size,
              'fields': 'accession,sequence,protein_name'}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get('results', [])
        ids, seqs = [], []
        for item in results:
            seq = item.get('sequence', {}).get('value', '')
            if seq and len(seq) >= 100:
                ids.append(item.get('primaryAccession', 'Unknown'))
                seqs.append(seq)
        return ids, seqs
    except Exception as e:
        print(f"  Error: {e}")
        return [], []

all_ids, all_seqs, all_labels = [], [], []
for cls, q in QUERIES.items():
    print(f"Fetching {cls}…")
    ids, seqs = fetch_uniprot(q)
    all_ids += ids; all_seqs += seqs; all_labels += [cls] * len(seqs)
    print(f"  → {len(seqs)} sequences")
    time.sleep(0.5)

df = pd.DataFrame({'protein_id': all_ids, 'sequence': all_seqs,
                   'enzyme_class': all_labels})
print(f"\nTotal sequences: {len(df)}")
print(df['enzyme_class'].value_counts())

le = LabelEncoder()
df['label'] = le.fit_transform(df['enzyme_class'])
CLASS_NAMES  = le.classes_.tolist()

# ─── 2. ESM-2 EMBEDDINGS ─────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
esm_model  = AutoModel.from_pretrained(model_name).to(device)
esm_model.eval()

def embed(seq: str) -> np.ndarray:
    inputs = tokenizer(seq, return_tensors='pt', padding=True,
                       truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = esm_model(**inputs)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()[0]

embeddings = np.array([embed(s) for s in tqdm(df['sequence'], desc="ESM-2")])
print(f"Embeddings shape: {embeddings.shape}")

# ─── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────
def simple_features(seq: str) -> np.ndarray:
    feats = [len(seq)]
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        feats.append(seq.count(aa) / len(seq))
    return np.array(feats, dtype=np.float32)

simple = np.array([simple_features(s) for s in df['sequence']])
X = np.hstack([embeddings, simple])
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 4. TRAIN RANDOM FOREST ──────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42,
                             n_jobs=-1, class_weight='balanced')
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_te)
print(f"\nTest accuracy : {accuracy_score(y_te, y_pred):.4f}")
print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

cv = cross_val_score(rf, X_scaled, y, cv=5, n_jobs=-1)
print(f"5-fold CV     : {cv.mean():.4f} ± {cv.std():.4f}")

# ─── 5. SAVE ARTIFACTS ───────────────────────────────────────────────────────
joblib.dump(rf,     'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\n✓ Saved rf_model.joblib and scaler.joblib")

# Also save class names so the app can verify alignment
import json
with open('class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f)
print("✓ Saved class_names.json:", CLASS_NAMES)
