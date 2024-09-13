from openai import OpenAI
import pandas as pd
secret_key = '' # insert OpenAI secret key
DAN = pd.read_excel("C:\\Users\\danie\\Documents\\tirocinio\\NLP Bcn\\Persona modulation\\DAN dataset.xlsx")
client = OpenAI(api_key = secret_key)

for i in range(len(DAN)):
    prompt = DAN.loc[i,'Prompt']
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-1106', #1106 is a model version with improved instruction following
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    DAN.loc[i,'Response'] = completion.choices[0].message.content

DAN.to_excel('DAN dataset completed.xlsx')