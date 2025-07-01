LONG_SUMMARY_NODE_PROMPT = """
You are a helpful news assistant. I will give you a list of article summaries or topic summaries. Please summarize all the summaries.

Here are the summaries:

<summaries>
{summaries}
</summaries>

Please summarize them into a single, 200 word summary, detailing the common themes and trends in the articles. Try to
synthesize the common themes and trends in the articles into a single summary.
""" 


SINGLE_SUMMARY_NODE_PROMPT = """
You are a helpful assistant. I will give you a summary. Please summarize it into a single, 2-5 word label.
Here is the summary:

<summary>
{summary}
</summary>

Please summarize it in a single, specific label that captures the meaning of the summary. 
Be precise and concise. Do not be too generic. Return just the label.

Your response:
"""