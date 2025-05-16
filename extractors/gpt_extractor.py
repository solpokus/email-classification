import openai

def extract_invoice_fields(email_body: str, api_key: str) -> str:
    openai.api_key = api_key
    prompt = f"""
Extract the following from the email:
- Invoice Number
- Invoice Date
- Total Amount

Email:
\"\"\"
{email_body}
\"\"\"
Answer in JSON format.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]
