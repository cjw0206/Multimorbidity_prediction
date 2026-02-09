# Multimorbidity Prediction via a Mixture-of-Experts Approach on a Patient-Disease Network

- **Problem:** Multimorbidity refers to the co-occurrence of multiple chronic conditions and can strongly affect a patient’s health course and treatment decisions.
- **Challenge:** The large variety of disease combinations and their irregular progression make accurate prediction difficult with standard statistical models or deep learning approaches.
- **Gap in prior work:** Many studies rely mainly on clinical variables from EHRs, limiting the use of non-clinical factors such as social and environmental conditions.

- **Data:** Health and Retirement Study (HRS)
- **Key idea:** Build a patient–disease heterogeneous network and formulate multimorbidity prediction as **link prediction** (patient ↔ disease).
- **Method:**
  - Use a **Graph Neural Network (GNN)** to learn node embeddings.
  - Use a **Mixture of Experts (MoE)** link predictor to model patient–disease interactions at the edge level.
  - The predictor has **four experts**, each combining node representations differently to capture complementary interaction patterns.

- **Results:**
  - Outperforms baseline methods; in the **stroke group**, improves **AUROC by ~5.3%** and **AUPRC by ~4.43%**.
  - Adding the **MoE** module improves performance by **~6.9%** compared with the backbone model.

- **Insights:** Ablation and visualization analyses suggest the four-expert design encourages **expert specialization** and contributes to better prediction.


## Model Architecture

<p align="center">
  <img src="Overall_workflow.png" alt="Overall Architecture" width="600"/>
</p>

<p align="left">
  <b>(a)</b> Gene Ontology(GO) and sequence embeddings from a protein pair are concatenated and encoded through a Transformer encoder with sparse MoE layers.  
  The encoded representations are element-wise multiplied, followed by a weighted attention and a linear layer for final prediction.<br>
  <b>(b)</b> GO embeddings are derived from Node2Vec trained on a random walk corpus over the GO graph.  
  Red symbols next to GO terms indicate proteins annotated with those terms (e.g., $P_1$, $P_2$).<br>
  <b>(c)</b> Protein sequences are tokenized, truncated, and encoded using ESM-2; the resulting embeddings are reshaped to match the GO embedding dimension.
</p>
