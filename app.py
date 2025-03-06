import streamlit as st
from lib import *
from func import embedding
@st.cache_resource
def load_model():
    # Không cache đối tượng không thể hash
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline)
    
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def find_file_in_directory(filename, directory):
    # Duyệt qua tất cả các file trong thư mục và các thư mục con
    for dirpath, dirnames, filenames in os.walk(directory):
        if filename in filenames:
            return os.path.join(dirpath, filename)  # Trả về đường dẫn tuyệt đối của file
    return None 
@st.cache_resource
def query(question, name_file):
    vector_db = Chroma(persist_directory= "data_embeded/all", embedding_function= embedding)
    # vector_db.get()
    # Tạo retriever để truy vấn dữ liệu
    # retriever = vector_db.as_retriever()
    directory = find_file_in_directory(name_file,"2024" )
    if directory == None:
        retriever = vector_db.as_retriever()
        print(1)
    else:
        retriever = vector_db.as_retriever(
        search_kwargs={"filter": {"source": {"$eq": directory}}})

    prompt = hub.pull("rlm/rag-prompt")
    llm = load_model()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    output = rag_chain.invoke(question)

    return output
def main():
    st.title("Search Your PDF")
    with st.expander("About the app"):
        st.markdown(
            """
            Heloooooooo testtttt nhaaaaa
            """
        )
    question = st.text_area("Enter Your Question")
    name_file = st.text_area("Enter your name file")
    if st.button("Search"):
        st.info("Your Question: " + question)
        if name_file != "":
            st.info("Name Your File: " + name_file)
        output = query(question, name_file)
        answer = output.split('Answer:')[1].strip()
        st.info("Your Answer: " + answer)


if __name__ == '__main__':
    main()
    
