from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)
