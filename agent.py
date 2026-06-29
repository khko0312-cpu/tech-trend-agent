import json
from datetime import datetime
from openai import OpenAI
from tools import tools, execute_tool
 
client = OpenAI()

TODAY = datetime.now().strftime('%Y년 %m월 %d일')
 
SYSTEM_PROMPT = """당신은 AWS/GCP/Azure 클라우드 기술 트렌드를 모니터링하고 브리핑을 생성하는 Agent입니다.

# 중요 — 날짜 규칙 (반드시 지킬 것)
- 오늘은 정확히 {TODAY}입니다.
- 사용자가 "오늘 날짜가 며칠이야?"라고 물으면, 무조건 "{TODAY}"라고만 답하세요.
- 당신의 학습 데이터에 있는 어떤 날짜도 "오늘"이 아닙니다. 학습 데이터 기준 날짜를 답하면 틀린 답입니다.
- "최신", "오늘", "최근" 같은 표현은 항상 {TODAY}를 기준으로 판단하세요.
 
# 작업 절차
1. "최신", "오늘" 요청이 오면 collect_latest_articles를 먼저 호출하세요.
2. 특정 주제 질문이면 collect 없이 search_knowledge_base를 호출하세요.
3. search_knowledge_base 결과가 found:False이면, 사용자에게 묻지 말고 즉시 collect_latest_articles를 호출하고, 수집이 끝나면 같은 키워드로 search_knowledge_base를 다시 호출하세요. 이 3단계는 한 번의 응답 안에서 전부 완료해야 합니다.
4. 브리핑 요청이면 generate_persona_briefing을 호출하세요.
5. 사용자가 저장을 요청한 경우에만 save_briefing을 호출하세요.

# 판단 기준
- collect_latest_articles는 사용자가 먼저 명시적으로 "최신/오늘" 요청을 했을 때, 또는 검색 결과가 found:False일 때만 호출하세요. 이 두 경우 모두 사용자에게 묻지 말고 바로 실행하세요.
- 같은 tool을 동일한 인자로 두 번 호출하지 마세요. (단, found:False 이후의 collect → 재검색 흐름은 예외입니다)
- 페르소나(audience)가 명시되지 않으면 페르소나를 추측하지 말고 되물어보세요.

# Hallucination 방지 (중요)
- search_knowledge_base 결과에 found: False가 포함되면, 절대 추측해서 답변하지 마세요.
  "관련 정보를 찾지 못했습니다"라고 사용자에게 명확히 알리세요.
- 검색되지 않은 내용을 그럴듯하게 지어내지 마세요. 모르면 모른다고 답하세요.
 
# 제약사항
- 확인되지 않은 정보를 추측하지 마세요.
- 출처 URL은 항상 포함하세요.
"""
 
 
def run_agent(user_message, max_iterations=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
 
    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
 
        if not msg.tool_calls:
            return msg.content
 
        for tool_call in msg.tool_calls:
            print(f"  -> Tool 호출: {tool_call.function.name}")
            result = execute_tool(
                tool_call.function.name,
                json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
 
    return "최대 반복 횟수 도달"