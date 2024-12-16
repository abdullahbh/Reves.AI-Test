# import os
# import os
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from bot import file_hash, load_and_process_local_pdf, create_rag_chain, api_key

# if __name__ == "__main__":
#     # Specify your local PDF file path
#     pdf_path = "company_profile.pdf"

#     # Directory to store the vectorstore
#     store_path = "my_vectorstore"
#     hash_path = os.path.join(store_path, "hash.txt")

#     # Get a hash of the PDF
#     pdf_hash = file_hash(pdf_path)

#     # Check if vectorstore already exists and if it's for the same PDF
#     if os.path.exists(store_path) and os.path.exists(hash_path):
#         with open(hash_path, 'r') as f:
#             saved_hash = f.read().strip()
#     else:
#         saved_hash = None

#     # Determine if we need to re-embed
#     if pdf_hash != saved_hash or not os.path.exists(store_path):
#         # If hash doesn't match or no stored vectorstore, re-embed
#         if not os.path.exists(store_path):
#             os.makedirs(store_path, exist_ok=True)
#         vectorstore = load_and_process_local_pdf(pdf_path)
#         vectorstore.save_local(store_path)
#         with open(hash_path, 'w') as f:
#             f.write(pdf_hash)
#     else:
#         # Load existing vectorstore
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key())
#         vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)


#     # Initialize the language model
#     llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key())

#     # Create the RetrievalQA chain
#     rag_chain = create_rag_chain(vectorstore, llm)

#     # Interactive Q&A loop
#     print("You can now ask questions about the document. Press Enter on an empty line to exit.")
#     while True:
#         user_question = input("\nEnter your question: ").strip()
#         if not user_question:
#             print("Exiting...")
#             break
#         answer = rag_chain.run(user_question)
#         print("Answer:", answer)