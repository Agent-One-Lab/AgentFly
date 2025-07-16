import openai

def test_think_prompt():
    client = openai.OpenAI(api_key="token-123", base_url="http://0.0.0.0:8000/v1")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A python code interpreter that can execute code and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The python code to execute"},
                    },
                },
            }
        }
    ]
    response = client.chat.completions.create(
        # model="/mnt/sharefs/users/haonan.li/models/Qwen2.5-7B-instruct-am_think_v1_distilled",
        model="Qwen/Qwen2.5-7B-Instruct",
        tools=tools,
        messages=[
            {"role": "user", "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."},
        ],
        max_tokens=8192
    )
    print(response.choices[0].message.content)