<div align="center">

# 🧬 ProteinOracle

**Enzyme Function Prediction · ESM-2 Embeddings · Explainable AI**

<br>

<a href="https://huggingface.co/spaces/somiya-khan01/protein-function-predictionn" target="_blank">
  <img src="https://img.shields.io/badge/🚀_LAUNCH_DEMO_-TRY_NOW-FF5722?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=000000&color=FF5722" alt="Launch Demo" width="350">
</a>

<br>


</div>

---

## 📋 About

**ProteinOracle** is a Streamlit-based web application that predicts enzyme function from an amino acid sequence. It leverages:

- **ESM-2** (Evolutionary Scale Modeling) protein language model embeddings
- **Random Forest** classifier trained on 198 real human enzyme sequences from UniProt
- **SHAP** (SHapley Additive exPlanations) for model interpretability

The model classifies sequences into **5 enzyme classes**:
- **Kinase** – Transfers phosphate groups
- **Phosphatase** – Removes phosphate groups
- **Protease** – Cleaves peptide bonds
- **Dehydrogenase** – Catalyzes oxidation-reduction reactions
- **Transferase** – Transfers functional groups



##  Features

| Feature | Description |
|---------|-------------|
| **Sequence Input** | Paste or upload amino acid sequence (FASTA format) |
| **Real-time Prediction** | Classifies enzyme class with confidence score |
| **Probability Distribution** | Bar chart showing probabilities for all 5 classes |
| **Explainability (SHAP)** | Highlights which ESM-2 features most influenced the prediction |
| **Sequence Statistics** | Length, amino acid composition, hydrophobicity, charge distribution |
| **Example Sequences** | Load pre-loaded examples (CDK2, Caspase-3, GAPDH, PP2A) |

##  Technical Pipeline
<img width="500" height="500" alt="ChatGPT Image Jun 9, 2026, 03_13_23 PM" src="https://github.com/user-attachments/assets/a71e426d-bb69-4226-8783-48e0a7015f22" />





## 🖥️ Dashboard Preview

### Prediction Interface
<img width="1527" height="573" alt="image" src="https://github.com/user-attachments/assets/5c14fb42-ac11-4ba7-a6ee-bddb4555034b" />


### Probability Distribution
<img width="1399" height="623" alt="image" src="https://github.com/user-attachments/assets/ebe9daa3-d50a-4b67-a29e-66247816b9b7" />


### Sequence Statistics
<img width="1461" height="618" alt="image" src="https://github.com/user-attachments/assets/46f0ed25-08fb-43cd-a5db-0eeacbe2e1cf" />




## Model Performance

### Dataset Composition (198 sequences)

| Enzyme Class | Number of Sequences |
|--------------|---------------------|
| Kinase | 40 |
| Phosphatase | 40 |
| Protease | 40 |
| Dehydrogenase | 39 |
| Transferase | 39 |

### Random Forest Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~88-92% |
| **5-fold CV Mean** | 0.89 |
| **5-fold CV Std** | 0.04 |

### Top Predictive Features (ESM-2 Embeddings)

| Feature | Importance (MDI) |
|---------|------------------|
| ESM_1030 | 0.016 |
| ESM_280 | 0.014 |
| ESM_243 | 0.013 |
| ESM_297 | 0.012 |
| ESM_1 | 0.011 |



##  Example Test Sequences

You can test the dashboard with these real human enzyme sequences:

| Enzyme | Class | Sequence (First 30 AA) |
|--------|-------|------------------------|
| **CDK2** | Kinase | `MENFQKVEKIGEGTYGVVYKARNKLTGEVV` |
| **Caspase-3** | Protease | `MENTENSVDSKSIKNLEPKIIHGSESMDSG` |
| **GAPDH** | Dehydrogenase | `MGKVKVGVNGFGRIGRLVTRAAFNSGKVD` |
| **PP2A** | Phosphatase | `MADAKELVSQFNEQIRRLDICERVLERPEN` |



