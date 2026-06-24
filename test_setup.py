from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=20,
    messages=[{"role": "user", "content": "안녕이라고만 대답해"}]
)
print(response.choices[0].message.content)