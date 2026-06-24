from vectorstore import collection, retrieve, rerank_by_recency

results = retrieve("Bedrock", top_k=5)
ranked = rerank_by_recency(results, query="Bedrock")

print("--- rerank_by_recency 결과 ---")
for doc, meta, score in ranked:
    print(f"score={score:.3f} | {meta['title']}")