import os
from fastapi import FastAPI
from pydantic import BaseModel
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured.ingest.runner import LocalRunner
from unstructured.staging.base import elements_from_json
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import utils as chromautils
from huggingface_hub import notebook_login
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA



"""=========================================================================================================================================="""
output_path = "./local-ingest-output"

# Create the directory
if not os.path.exists(output_path):
    os.makedirs(output_path)  # Use makedirs to create any intermediate directories
    print(f'Directory "{output_path}" created.')
else:
    print(f'Directory "{output_path}" already exists.')


# runner = LocalRunner(
#     processor_config=ProcessorConfig(
#         # logs verbosity
#         verbose=True,
#         # the local directory to store outputs
#         output_dir=output_path,
#         num_processes=2,
#     ),
#     read_config=ReadConfig(),
#     partition_config=PartitionConfig(
#         partition_by_api=True,
#         api_key="9q8hHHdxfarMRXZvhOm2vMo72apT80",
#     ),
#     connector_config=SimpleLocalConfig(
#         input_path="harry_potter.pdf",
#         # whether to get the documents recursively from given directory
#         recursive=False,
#     ),
# )
# runner.run()

elements = []
for filename in os.listdir(output_path):
    filepath = os.path.join(output_path, filename)
    elements.extend(elements_from_json(filepath))

chunked_elements = chunk_by_title(
    elements,
    max_characters=512,
    combine_text_under_n_chars=200,
)

documents = []
for chunked_element in chunked_elements:
    metadata = chunked_element.metadata.to_dict()
    metadata["source"] = metadata["filename"]
    del metadata["languages"]
    documents.append(Document(page_content=chunked_element.text, metadata=metadata))


"""=========================================================================================================================================="""
# ChromaDB doesn't support complex metadata, e.g. lists, so we drop it here.
# If you're using a different vector store, you may not need to do this
# docs = chromautils.filter_complex_metadata(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# vectorstore = Chroma.from_documents(documents, embeddings)
# create embeddings and DB
vectorstore = FAISS.from_documents(documents = documents, embedding = embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


"""=========================================================================================================================================="""

notebook_login()


"""=========================================================================================================================================="""

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=200,
    eos_token_id=terminators,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions using provided context.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

# question = "summarize chapter six"

# qa_chain.invoke(question)["result"]


"""=========================================================================================================================================="""
# building api server with FastAPI
app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/generate")
def generate(query: Query):
  try:
    answer = qa_chain.invoke(query.query)["result"]
    return {"Answer": answer}
  except Exception as e:
    return {"Error": str(e)}

# To run the server, use: uvicorn your_script_name:app --host 0.0.0.0 --port 8000
