import openai
import os

os.environ['OPENAI_API_KEY'] = open('/Users/spangher/.openai-bloomberg-project-key.txt').read().strip()
client = openai.OpenAI()

# Function to count all descendants (entire subtree) of a node
def count_descendants(G, node):
    return 1 + sum(count_descendants(G, child) for child in G.successors(node))

def call_openai(prompt):
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message.content

