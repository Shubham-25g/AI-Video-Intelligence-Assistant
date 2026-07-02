import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def format_chat_history(chat_history_list):
    """Formats the running conversation list into a coherent block for systemic memory context."""
    formatted_history = ""
    for msg in chat_history_list:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history or "No conversational history yet."

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    
    # UPGRADE: Passing full transcript context down to unlock Hybrid Retrieval execution paths
    retriever = get_retriever(vector_store, transcript=transcript, k=6)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below. You are also given the current 
conversation history so you can seamlessly handle sequential follow-up questions.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Current Conversation History:
{chat_history}

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "chat_history": RunnableLambda(lambda x: format_chat_history(x["chat_history"])),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str, chat_history: list = None):
    """UPGRADE: Uses .stream() instead of .invoke() to stream raw text output tokens back dynamically."""
    return rag_chain.stream({
        "question": question, 
        "chat_history": chat_history or []
    })