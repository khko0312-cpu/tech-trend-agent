import os
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
 
load_dotenv()
 
chroma_client = chromadb.PersistentClient(path="./chroma_db")
 
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
 
collection = chroma_client.get_or_create_collection(
    name="tech_trends",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)
 
RECENCY_TRIGGER_WORDS = ["최신", "오늘", "이번 주", "최근", "방금"]
 
 
def store_article(article, analysis, chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{article['id']}_{i}"],
            documents=[chunk],
            metadatas=[{
                "source":     article["source"],
                "title":      article["title"],
                "url":        article["url"],
                "published":  article["published"],
                "category":   analysis["category"],
                "importance": analysis["importance"],
                "keywords":   ",".join(analysis["keywords"]),
            }]
        )
 
 
def retrieve(query, filters=None, top_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=filters if filters else None,
        include=["documents", "metadatas", "distances"]
    )
    return results
 
 
def get_recency_weight(query):
    """질문 유형에 따라 최신성 가중치를 동적으로 조정.
    '최신' 류 표현이 있으면 최신성을 강하게 반영하고,
    그 외에는 유사도를 더 신뢰해 오래된 문서도 동등하게 평가."""
    if any(kw in query for kw in RECENCY_TRIGGER_WORDS):
        return 0.5
    return 0.15
 
 
def rerank_by_recency(results, query="", recency_weight=None):
    if recency_weight is None:
        recency_weight = get_recency_weight(query)
 
    if not results["documents"][0]:
        return []
 
    docs = zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    scored = []
    for doc, meta, dist in docs:
        similarity_score = 1 - dist
        try:
            pub = datetime.fromisoformat(meta["published"][:10])
            days_old = (datetime.now() - pub).days
        except Exception:
            days_old = 30
        # 오래된 문서를 배제하지 않고, 점수만 완만하게 낮춤 (하드 필터링 없음)
        recency_score = max(0, 1 - days_old / 30)
        final_score = (1 - recency_weight) * similarity_score + recency_weight * recency_score
        scored.append((doc, meta, final_score))
    return sorted(scored, key=lambda x: x[2], reverse=True)
 
 
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
 
    from collectors import collect_all
    from preprocess import deduplicate, analyze_article, chunk_text
 
    articles = deduplicate(collect_all(limit_per_source=5))
 
    for a in articles:
        analysis = analyze_article(a)
        if analysis:
            chunks = chunk_text(a["summary"] or a["title"])
            store_article(a, analysis, chunks)
            print(f"저장 완료: [{a['source']}] {a['title'][:40]}")
 
    # 검색 테스트 - "최신" 키워드 있는 질문 (recency_weight 높게 적용)
    query1 = "최신 AWS Lambda 소식"
    results1 = retrieve(query1, top_k=3)
    ranked1 = rerank_by_recency(results1, query=query1)
    print(f"\n[질문: '{query1}' - recency_weight={get_recency_weight(query1)}]")
    for doc, meta, score in ranked1:
        print(f"  [{score:.2f}] {meta['title']}")
 
    # 검색 테스트 - 일반 질문 (유사도 중심)
    query2 = "Lambda 콜드 스타트 해결 방법"
    results2 = retrieve(query2, top_k=3)
    ranked2 = rerank_by_recency(results2, query=query2)
    print(f"\n[질문: '{query2}' - recency_weight={get_recency_weight(query2)}]")
    for doc, meta, score in ranked2:
        print(f"  [{score:.2f}] {meta['title']}")