import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots"
              "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
text_summary = pipeline("summarization", model = model_path,
                dtype = torch.bfloat16)


#text =''' Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of January 2026, Forbes estimates his net worth to be around US$788 billion'''
#print(text_summary(text));

def summary (inputs):
    output = text_summary(inputs)
    return output[0]["summary_text"]

gr.close_all()

#demo = gr.Interface(fn = summary, inputs = "text", outputs = "text")
demo = gr.Interface(fn = summary,
                    inputs = [gr.Textbox(label = "Input text to summarize", lines = 6)],
                    outputs = [gr.Textbox(label = "Summarized Text", lines = 4)],
                    title = "Text Summarizer",
                    description = " THIS APPLICATION IS USED TO SUMMARIZE THE TEXT")

demo.launch(share = True)
# if we write 'share = True' in 'launch()' then demo.launch(share=True)
# Gradio creates a public URL, something like:
#
# https://abcd-1234.gradio.live
# Now anyone can open your app from anywhere in the world using that link.