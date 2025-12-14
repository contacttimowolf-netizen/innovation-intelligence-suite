# Evaluation Metrics

In real world, the product development would feature iterative testings with the target audience and sprints to refine selection of connected data sources, best data processing and prompt layer.

### RAG Model

The performance of the model was evaluated using three methods:

1. Qualitative Test Questions

We defined seven representative user questions, which can be found in 06_doc/user_queries.md.
The model’s responses were reviewed through an iterative process in which source documents were refined and the prompt template was adjusted until satisfactory accuracy and clarity were achieved.
The resulting outputs from the last iteration is available in 07_testsdemo/test_outputs.

2. Relevance Score

A relevance score measures how well the retrieved documents align semantically with the user query. In our system, this score is calculated using FAISS’s cosine-similarity distance metric, which compares the query embedding with document embeddings. Cosine similarity returns a value between –1 and 1, with higher values indicating stronger semantic alignment.

For this project, out goal was to have a FAISS cosine-similarity score of 0.5 or higher.

The threshold set for the model was a similarity score of 0.3 to ensure that only the most semantically relevant documents were passed to the model for generation (The score was reduced to 0.15 for startup questions and 0.25 for patent questions as it was found these questions needed boosting).

3. Loading Time

We set a performance target requiring the system to load and respond in under 10 seconds, including document retrieval, model inference, and any post-processing steps.

### Predictive Model

Semantic Coherence & Seed Orthogonality Evaluation & Topic Modelling

To evaluate the semantic coherence of the technology classification and the quality of the seed-based ontology, we relied on silhouette score as the primary quantitative metric. The silhouette score measures how well documents assigned to the same technology cluster are separated from documents assigned to other clusters, providing an intrinsic assessment of clustering quality in the embedding space.

A target silhouette score of 0.5 was defined as an indicator of strong semantic separation. While this value was treated as an aspirational benchmark rather than a strict acceptance criterion, it served as a reference point for evaluating different assignment and clustering configurations.

In addition to quantitative evaluation, sanity checks were performed using cosine similarity distributions between documents and their assigned technology seeds. Specifically, we analysed histogram shapes to assess seed orthogonality. A single-mode (unimodal) distribution was expected and observed, indicating that technology seeds were not competing for the same semantic regions and that assignments were not driven by ambiguous overlaps between seed definitions.

The absence of multimodal or heavy-tailed distributions provided evidence that the seed ontology formed a semantically consistent and stable partitioning of the document space, rather than a fragmented or overlapping classification.

This combined approach—using silhouette score for structural separation and similarity histograms for orthogonality and confidence diagnostics—ensured that downstream trend analysis and forecasting were built on a robust and interpretable semantic foundation, without relying on hard similarity thresholds or heuristic filtering.




