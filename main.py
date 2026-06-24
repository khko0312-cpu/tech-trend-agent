import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent import run_agent

if __name__ == "__main__":
    print("기술 트렌드 모니터링 Agent (종료: exit 입력)\n")
    while True:
        user_input = input("질문: ")
        if user_input.strip().lower() == "exit":
            break
        result = run_agent(user_input)
        print("\n결과:\n", result, "\n")