# llama3-RAG
This is an implementation of  LLAMA3 with RAG using Ollama, ChromaDB and Flask API with PDF upload and search functionality.

The original implementation of this project is available at https://github.com/ThomasJay/RAG/tree/main and you can find the youtube video explaining the process step by step at https://youtu.be/7VAs22LC7WE?si=dJZmQimjKoDujtSQ . 

This project has helped me to gain familiarity to the concepts of Retreival Augmented Generation (RAG),langchain,  Ollama, Chromadb and Flask API. I have also come to about one of the use cases of Postman. 

<h2> Breakdown of Implementation of the program </h2>

1. Download ollama and access LLMs like llama3.
2. Start the Ollama server using command "ollama serve"
3. Build NLP pipeline using Langchain for tasks such as document parsing, embedding, text splitters and model interaction. 
5. Create Flask API endpoint that handles HTTP POST requests to the "/ask_pdf" route.
6. Using POSTMAN to POST HTTP requests and get answers from the LLAMA3.

<h2> Future application </h2>

- I am going to build a PDF Question answering system using DEEPSEEK R1, which will have multiple document upload and search capabilities and i will also design  frontend using __Streamlit__ library and integrate both the frontend and backend to create my own QA system that could run locally.
