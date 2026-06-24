import json
from openai import OpenAI
from tools import tools, execute_tool
 
client = OpenAI()
 
SYSTEM_PROMPT = """당신은 AWS/GCP/Azure 클라우드 기술 트렌드를 모니터링하고 브리핑을 생성하는 Agent입니다.
 
# 작업 절차
1. "최신", "오늘" 요청이 오면 collect_latest_articles를 먼저 호출하세요.
2. 특정 주제 질문이면 collect 없이 search_knowledge_base를 호출하세요.
3. 브리핑 요청이면 generate_persona_briefing을 호출하세요.
4. 사용자가 저장을 요청한 경우에만 save_briefing을 호출하세요.
 
# 판단 기준
- 같은 tool을 동일한 인자로 두 번 호출하지 마세요.
- 검색 결과가 없으면 collect_latest_articles로 먼저 데이터를 수집하세요.
- 페르소나(audience)가 명시되지 않으면 developer를 기본값으로 사용하세요.
- collect_latest_articles는 한 번의 대화에서 최대 1회만 호출하세요. 이미 호출했다면 그 결과를 그대로 사용하세요.
- Tool 실행 결과가 성공(stored > 0)이면, 최종 답변에서 "수집하지 못했다"고 말하지 마세요. 실제 Tool 결과를 정확히 반영해서 답변하세요.
 
# Hallucination 방지 (중요)
- search_knowledge_base 결과에 found: False가 포함되면, 절대 추측해서 답변하지 마세요.
  "관련 정보를 찾지 못했습니다"라고 사용자에게 명확히 알리세요.
- 검색되지 않은 내용을 그럴듯하게 지어내지 마세요. 모르면 모른다고 답하세요.
 
# 제약사항
- 확인되지 않은 정보를 추측하지 마세요.
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