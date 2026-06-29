from dotenv import load_dotenv
load_dotenv()

import httpx
import json
from bs4 import BeautifulSoup
from openai import OpenAI
 
client = OpenAI()
seen_ids = set()
 
 
def deduplicate(articles):
    unique = []
    for a in articles:
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            unique.append(a)
    return unique
 
 
def extract_body(url):
    try:
        html = httpx.get(url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        print(f"본문 추출 실패: {e}")
        return ""
 
 
def analyze_article(article, retries=3):
    body = extract_body(article["url"]) or article["summary"]

    prompt = f"""다음 기술 아티클을 분석하세요.
제목: {article['title']}
본문: {body}

아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
  "category": "AI/ML | Infrastructure | Security | Database | DevOps | Other",
  "importance": 1~5,
  "summary_tech": "개발자용 2줄 요약",
  "summary_biz": "영업용 2줄 요약",
  "summary_exec": "경영진용 1줄 요약",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}}"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            with open("error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- {article['title']} ---\n")
                f.write(f"에러 타입: {type(e).__name__}\n")
                f.write(f"에러 내용: {e}\n")
            continue
    return None
 
 
def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks
 
 
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
 
    from collectors import collect_all
 
    articles = deduplicate(collect_all(limit_per_source=5))
    results = []
    for a in articles:
        analysis = analyze_article(a)
        if analysis:
            results.append({**a, "analysis": analysis})
            print(f"[OK] [{a['source']}] {a['title'][:40]} -> {analysis['category']} (중요도 {analysis['importance']})")
        else:
            print(f"[FAIL] 분석 실패: {a['title'][:40]}")
 
    print(f"\n총 {len(results)}/{len(articles)}건 분석 완료")