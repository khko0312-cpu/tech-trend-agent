# evaluate.py
import json
from datetime import datetime
from openai import OpenAI
from vectorstore import retrieve, rerank_by_recency

client = OpenAI()


def evaluate_search_relevance(query, top_k=5):
    """검색 정확도 평가: 검색된 문서가 질문과 실제로 관련 있는지 LLM이 채점"""
    results = retrieve(query, top_k=top_k)
    ranked = rerank_by_recency(results, query=query)

    if not ranked:
        return {"query": query, "found": False, "relevance_scores": []}

    scores = []
    for doc, meta, score in ranked:
        prompt = f"""질문: {query}
문서 제목: {meta['title']}
문서 내용 일부: {doc[:300]}

이 문서가 질문과 관련이 있는지 1~5점으로 평가하세요 (5점: 매우 관련, 1점: 전혀 무관).
숫자만 답하세요."""
        response = client.chat.completions.create(
            model="gpt-4o", max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            relevance = int(response.choices[0].message.content.strip())
        except ValueError:
            relevance = None
        scores.append({"title": meta["title"], "score": relevance})

    return {"query": query, "found": True, "relevance_scores": scores}


def evaluate_hallucination_guard(query_without_data):
    """Hallucination 방지 평가: DB에 없을 질문에 found:False가 정확히 나오는지 확인"""
    from tools import execute_tool
    result = execute_tool("search_knowledge_base", {"query": query_without_data})
    passed = result.get("found") == False
    return {"query": query_without_data, "found_false_correctly": passed, "raw_result": result}


def evaluate_persona_tone(briefing_text, audience):
    """페르소나 적합도 평가: 브리핑이 해당 페르소나 톤을 지키는지 LLM이 채점"""
    criteria = {
        "developer": "기술 스펙, API, 성능 수치 등 기술적 디테일이 충분한가",
        "sales": "비즈니스 임팩트와 고객 설득 포인트가 중심인가",
        "executive": "간결한 요약과 액션 아이템이 포함되어 있는가",
    }
    prompt = f"""다음 브리핑이 '{audience}' 페르소나에 적합한지 평가하세요.
평가 기준: {criteria.get(audience, '')}

브리핑:
{briefing_text}

1~5점으로만 답하세요 (5점: 매우 적합, 1점: 전혀 부적합)."""
    response = client.chat.completions.create(
        model="gpt-4o", max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        score = int(response.choices[0].message.content.strip())
    except ValueError:
        score = None
    return {"audience": audience, "tone_fit_score": score}


def run_evaluation_suite():
    """전체 평가 실행 + 결과를 파일로 저장"""
    report = {
        "evaluated_at": datetime.now().isoformat(),
        "search_relevance": [
            evaluate_search_relevance("Bedrock 최신 소식"),
            evaluate_search_relevance("Lambda 콜드 스타트"),
        ],
        "hallucination_guard": [
            evaluate_hallucination_guard("화성 탐사 로봇 기술 트렌드"),
        ],
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("평가 완료 → evaluation_report.json 저장됨")
    return report


if __name__ == "__main__":
    run_evaluation_suite()