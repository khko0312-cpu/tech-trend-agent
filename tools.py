import json
import os
from datetime import datetime
from openai import OpenAI
 
from collectors import collect_all
from preprocess import deduplicate, analyze_article, chunk_text
from vectorstore import store_article, retrieve, rerank_by_recency
 
client = OpenAI()
 
tools = [
    {
        "type": "function",
        "function": {
            "name": "collect_latest_articles",
            "description": "AWS/GCP/Azure Blog, Hacker News에서 최신 기술 아티클을 수집한다. 사용자가 '최신 소식', '오늘 뉴스' 등을 요청할 때 사용.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "이미 저장된 기술 트렌드 문서에서 관련 내용을 검색한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "검색 질의문 (사용자의 원래 질문 그대로)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_persona_briefing",
            "description": "검색된 아티클들로 지정한 페르소나용 브리핑을 생성한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "브리핑 주제"},
                    "audience": {
                        "type": "string",
                        "enum": ["developer", "sales", "executive"],
                        "description": "브리핑 대상. 명시 없으면 developer 사용."
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_briefing",
            "description": "생성된 브리핑을 markdown 파일로 저장한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "저장할 브리핑 내용"}
                },
                "required": ["content"]
            }
        }
    }
]
 
STYLE_MAP = {
    "developer": "기술 스펙, API 변경사항, 성능 수치를 중심으로",
    "sales":     "비즈니스 임팩트와 고객 설득 포인트를 중심으로",
    "executive": (
        "3줄 이내 핵심 요약 후, 다음 구조로 작성: "
        "시장 시사점(이 변화가 업계 흐름에서 의미하는 것), "
        "경쟁 영향(경쟁사 대비 우리 위치에 미치는 영향), "
        "검토 포인트(지금 검토해볼 만한 질문이나 의사결정 1~2개). "
        "확정적인 '전략'이 아니라 '검토해볼 포인트'로 제시하고, "
        "참고 아티클에 근거하지 않은 추측은 하지 마세요."
    ),
}
 
 
def extract_keywords(query):
    """질문에서 검색용 키워드를 추출하는 전처리 노드 (피드백 1번)"""
    prompt = f"""다음 질문에서 검색에 사용할 핵심 키워드를 추출하세요.
질문: {query}

쉼표로 구분된 키워드만 응답하세요 (설명 없이). 2~4개 정도가 적당합니다.
고유명사(서비스명 등)는 영어 그대로, 일반 명사는 한국어로 추출하세요.
예: "Bedrock이랑 SageMaker 최신 업데이트 비교해줘" -> Bedrock, SageMaker, 업데이트
예: "GCP Vertex AI 트러블슈팅 방법" -> GCP, Vertex AI, 트러블슈팅"""
 
    try:
        response = client.chat.completions.create(
            model="gpt-4o", max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        keywords = [k.strip() for k in response.choices[0].message.content.split(",") if k.strip()]
        return keywords if keywords else [query]
    except Exception as e:
        print(f"키워드 추출 실패, 원본 질문으로 대체: {e}")
        return [query]
 
 
def execute_tool(name, args):
    if name == "collect_latest_articles":
        articles = deduplicate(collect_all(limit_per_source=10))
        stored = 0
        for a in articles:
            analysis = analyze_article(a)
            if analysis:
                chunks = chunk_text(a["summary"] or a["title"])
                store_article(a, analysis, chunks)
                stored += 1
        return {"collected": len(articles), "stored": stored}
 
    elif name == "search_knowledge_base":
        query = args["query"]
        keywords = extract_keywords(query)
        print(f"  ㄴ 추출된 키워드: {keywords}")
 
        all_results = []
        for kw in keywords:
            results = retrieve(kw, top_k=5)
            ranked = rerank_by_recency(results, query=query)
            all_results.extend(ranked)
 
        # 중복 제거 (여러 키워드에서 같은 문서가 검색될 수 있음)
        seen = set()
        unique_results = []
        for doc, meta, score in sorted(all_results, key=lambda x: x[2], reverse=True):
            if meta["url"] not in seen and score >= 0.25:
                seen.add(meta["url"])
                unique_results.append((doc, meta, score))
 
        # Hallucination 방지: 결과 없으면 명시적으로 알림
        if not unique_results:
            return {"found": False, "message": "관련 문서를 찾지 못했습니다."}
 
        return {
            "found": True,
            "results": [
                {"title": m["title"], "url": m["url"], "category": m["category"]}
                for _, m, _ in unique_results[:5]
            ]
        }
 
    elif name == "generate_persona_briefing":
        audience = args.get("audience", "developer")
        query = args["topic"]
        keywords = extract_keywords(query)
 
        all_results = []
        for kw in keywords:
            results = retrieve(kw, top_k=5)
            ranked = rerank_by_recency(results, query=query)
            all_results.extend(ranked)
 
        seen = set()
        unique_results = []
        for doc, meta, score in sorted(all_results, key=lambda x: x[2], reverse=True):
            if meta["url"] not in seen:
                seen.add(meta["url"])
                unique_results.append((doc, meta, score))
 
        if not unique_results:
            return {"briefing": f"'{query}'에 대한 관련 문서를 찾지 못했습니다. 먼저 데이터를 수집해주세요."}
 
        context = "\n".join([f"- {m['title']} ({m['url']})" for _, m, _ in unique_results[:5]])
 
        prompt = f"""다음 아티클들을 바탕으로 {audience}용 브리핑을 작성하세요.
주제: {query}
참고 아티클:
{context}

{STYLE_MAP.get(audience, STYLE_MAP['developer'])} 마크다운으로 작성하세요.
주의: 참고 아티클에 없는 내용은 추측해서 작성하지 마세요.

브리핑 작성 후, 마지막에 "## 추가로 알아보면 좋을 질문" 섹션을 만들어
위 참고 아티클 내용에 근거해서 사용자가 다음에 물어볼 만한 후속 질문 3개를 제안하세요.
참고 아티클에 없는 내용을 추측해서 질문을 만들지 마세요."""
 
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"briefing": response.choices[0].message.content}
 
    elif name == "save_briefing":
        os.makedirs("briefings", exist_ok=True)
        filename = f"briefings/briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(args["content"])
        return {"saved_to": filename}
 
    return {"error": f"알 수 없는 tool: {name}"}