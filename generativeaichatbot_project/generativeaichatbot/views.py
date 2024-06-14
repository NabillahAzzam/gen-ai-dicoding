from django.shortcuts import render
from django.http import JsonResponse
import pymilvus
import settings
from pathlib import Path
from pymilvus import connections
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Create your views here.
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .rag_model import RagModel
# from .milvus_utils import connect_milvus, create_collection

# class ChatbotAPIView(APIView):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model = RagModel()
#         connect_milvus()
#         self.collection = create_collection()

#     def post(self, request, *args, **kwargs):
#         input_text = request.data.get("message", "")
#         if input_text:
#             response_text = self.model.generate_response(input_text)
#             return Response({"response": response_text}, status=status.HTTP_200_OK)
#         return Response({"error": "No input message provided"}, status=status.HTTP_400_BAD_REQUEST)
os.environ['HUGGINGFACE_HUB_TOKEN'] = settings.HUGGINGFACE_HUB_TOKEN

connections.connect(
    alias="default",
    host='localhost',
    port='19530'
)


# Set up the model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_config = transformers.AutoConfig.from_pretrained(
   model_id,
)

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, config=model_config, device_map='auto', quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

generate_text = pipeline(model=model, tokenizer=tokenizer, task="text-generation", return_full_text=True, temperature=0.9, max_length=1024, do_sample=True, pad_token_id=tokenizer.eos_token_id)

def generate_module(user_input):
    template = f"""
    Create a learning module based on the following user input: {user_input}.
    Explains the learning module completely and is easy to understand.
    If you don't know the answer, just say that you don't know, don't try to make up an answer and don't produce repetitive answers.
    Make sure your answer in Indonesian
    Answer :
    """
    result = generate_text(template, max_length=1024, num_return_sequences=1)
    generated_text = result[0]['generated_text']
    answer_start = generated_text.find("Answer :")
    answers = generated_text[answer_start:].strip()
    return answers

def generate_quiz(quiz_context):
    prompt_assessment  = """
    You are an expert quizzes maker for technical fields. Let's think step by step and
    create a quizzes with multiple choice questions about the following concept/content: {quiz_context}.
    Answer according to the quiz format and make sure your answer is in Indonesian.
    Make sure don't produce repetitive answers.
    """
    final_prompt = prompt_assessment.format(quiz_context=quiz_context)
    result = generate_text(final_prompt, max_length=1024, num_return_sequences=1)
    generated_text = result[0]['generated_text']
    answer_start = generated_text.find("Answer :")
    answers = generated_text[answer_start:].strip()
    return answers

def generate_chatbot(user_input):
    PROMPT_TEMPLATE =  """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer and don't produce repetitive answers.
    Make sure not to create any other questions aside from {user_input}
    Make sure your answer in Indonesian
    {context}
    Question: {user_input}
    Answer :
    """
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "user_input"])
    output_parser = StrOutputParser()
    retriever = Milvus.as_retriever()
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),"user_input": RunnablePassthrough()}
        | prompt
        | generate_text
        | StrOutputParser()
    )
    answer = rag_chain.invoke(user_input)
    answer_start = answer.find("Answer :")
    final_answer = answer[answer_start:].strip()
    return final_answer


def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        if "materi" in message:
            response = generate_module(message)
        elif "soal" in message:
            response = generate_quiz(message)
        else:
            response = generate_chatbot(message)
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')

