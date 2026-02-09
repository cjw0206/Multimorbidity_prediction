# Multimorbidity Prediction via a Mixture-of-Experts Approach on a Patient-Disease Network


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
