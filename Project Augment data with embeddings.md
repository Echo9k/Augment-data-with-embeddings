[TOC]



# Project overview

Improve companies interaction with our knowledge bases. InfoHub seeks to utilize groundbreaking large language models to deliver a system whereby a user’s questions about company data can be answered through a Q&A-style language interface. You’ll begin by assisting them in creating this tool by processing, tokenizing, and converting data into embeddings with BeautifulSoup and tiktoken.

# 1. Augment data with embeddings

## 1. Prepare the Data and Split It

### **Objective**

Prepare the data by extracting it, parsing it, and splitting it into chunks.

### Workflow:

Step 1:

Create a Python function to **extract text from HTML** files using Beautiful Soup. Keep in mind:

- Reading Files: Read files from paths, handling various character encodings.
- HTML Parsing: Use Beautiful Soup to parse the HTML. Make sure to initialize your “soup” object properly.
- Text Extraction: Focus on extracting plain text, excluding scripts, styles, or tags.
- Return Value: Ensure the function returns the clean text.

Step 2:

Next, create a function to **segment texts by converting text to tokens** for AI processing using the tiktoken library. Your guide includes:

- Chunking Text: Split the text into manageable chunks, respecting the maximum token count, not just the word count.
- Iterative Process: Check if adding a word exceeds the token limit. If so, start a new chunk; otherwise, keep appending.
- Finalizing Chunks: Include the leftover text as the final chunk post-iteration.

Step 3:

Combine both functions into one.



### **Deliverable**

The deliverable for this milestone is a set of Python functions:

- The final result should be a **list of chunks of text.**
- A function that extracts text from the HTML dataset using Beautiful Soup
- A function that **splits your text field into discrete chunks using tiktoken**, so that later we can calculate their embeddings.
- A function that combines both of these functions into one.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

## **2. Calculate Embeddings for Text chunks and Add Them to the Dataframe**

### **Objective**

Calculate embeddings for the text chunks and insert them into a DataFrame.

### **Importance to project**

Embeddings play a pivotal role in enhancing the capabilities of Large Language Models. They enable LLMs to grasp and analyze the underlying meanings in data. This capability is an aid to executing precise semantic searches, connecting related concepts seamlessly.

### **Workflow**

Step 1:

Convert the text into numerical embeddings, capturing its semantic essence for machine comprehension. Key points:

- Understanding Embeddings: Text embeddings transform the text into numerical vectors, which is useful in machine learning.
- Leveraging OpenAI: **Use the OpenAI API to create the embeddings.** Understand the specifics of API requests for text chunks.
- Batch Processing: Opt for sending text in batches for efficiency. Be mindful of not overloading the API or surpassing batch limits.
- Storing Results: Systematically save the embeddings from each batch, aiming to compile all of them by the function’s end.
- Monitoring Progress: Maintain a log of the embedding generation stages, particularly for large data sets.

Step 2:

Convert the raw HTML files into organized data with text content and embeddings. This process includes:

- Directory Navigation: Probe a directory for HTML files, filtering out unrelated file types.
- Text Extraction & Tokenization: Extract and tokenize text from each HTML file, adhering to token limits.
- Batched Embedding Calculation: Convert text chunks into embeddings using your batch-processing function.
- Data Structuring: Track the relationship between embeddings, text chunks, and their original files.
- Final Aggregation: Post-processing, consolidate the data into a DataFrame, detailing file paths, text segments, and embeddings.

#### **Deliverable**

The deliverable for this milestone is a Python program that contains the following functions:

- A function that calculates embeddings using an LLM model.
- A function that combines the functions we created in Milestone 1 and extracts text, chunks it, and calculates their embeddings, outputting a dictionary.
- A function that converts that dictionary into a DataFrame.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

## **3. Use Similarity Search to Select Text and Query the LLM**

### **Objective**

Create a function that **performs similarity search based on embeddings** and use it to create a **prompt that combines the extracted text** and asks the LLM a question.

### **Importance to project**

Once we have the embeddings, we can use them to augment the knowledge of our LLM by inserting them into the prompt. This will radically improve the quality of the answers.

### **Workflow**

Step 1:

Create a function to pinpoint strings closely matching a search term, which is vital in various natural language applications. Here’s a summary:

- Embedding the search term: Use the OpenAI API to create embeddings for a search term to form the basis for similarity searches.
- Exploring similarity metrics: Learn about cosine similarity to gauge the likeness between vectors, ranging from -1 to 1.
- Comparison loop: Compare the search term’s embedding with the others in your DataFrame, calculating the similarity scores.
- Curate the results: Sort your findings based on similarity, returning only the top ‘limit’ entries.
- Output the specs: Return a tuple with two lists: the strings and their similarity scores.

Step 2:

Blend token counting with semantic search to create query-based messages. Here’s your guide:

- Token mastery: Understand token counting, which is essential for managing API interactions.
- Utilize past functions: Use `search_similar_strings` to find strings in the DataFrame that resemble a user’s query.
- Message assembly: Formulate a clear message combining an intro, a user query, and the pertinent text.
- Output design: Generate a single, coherent message for subsequent model processing.

Step 3:

In this final step, you will unify all of the previous elements to make queries to the GPT model. Your roadmap:

- Craft messages: Use `query_message` to create user-query-based messages, pulling the relevant texts.
- GPT interaction: Provide a context sequence to the GPT model and capture its reply.
- Handle the output: Optionally, display and check the message to ensure the function outputs the model’s answer.

### **Deliverable**

The deliverable for this milestone is a Python file that contains the following:

- A function that performs a similarity search between the user query and the embeddings in the DataFrame.
- A function that dynamically constructs a user message while converting its query into an embedding, performs a similarity search, and later inserts the search result into the prompt.
- A function that integrates the others and sends a question to the LLM.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

## Project Conclusions

You’ve undertaken the first step in enhancing Large Language Models (LLMs) Prepare the Data and Split it using LangChainthrough document retrieval techniques. This accomplishment has resulted in the development of a set of functions that convert various forms of data into embeddings. These functions significantly expand the capacity of LLMs, allowing them to interact with and learn from data beyond their initial training sets.

This achievement is not merely about improving efficiency; it’s a strategic enhancement of your LLMs’ analytical capabilities. By transforming raw data into embeddings, you have equipped the model to handle a broader range of inquiries with increased precision. This advancement highlights the potential for integrating more complex, real-world data into AI systems and demonstrates what’s possible in the field of machine learning.

Looking ahead, this project serves as a solid foundation for further advancements in the realm of artificial intelligence. The methodologies you’ve employed and the systems you’ve developed open up new avenues for exploration. This progress is a gateway to continued innovation, providing the groundwork necessary to explore more sophisticated data-retrieval mechanisms, enhance algorithmic efficiency, and expand the overall scope and adaptability of Large Language Models.

# 2: Q&A Using Vector Databases

## Techniques employed

- LangChain framework: A platform that simplifies the integration of LLMs into various applications like document analysis and chatbot creation.
- Chroma vector store: Enables efficient storage and management of high-dimensional data, such as embeddings, to facilitate fast data retrieval.
- OpenAIEmbeddings: Converts textual data into numerical embeddings using OpenAI’s large language models.
- API key management: Best practices for securely handling API keys using environment variables.
- TQDM library: Provides progress bars for real-time feedback during iterative processes.
- RetrievalQA: Combines language models with vectorized data to answer user queries quickly and accurately.

## 2.1 Prepare the Data and Split it using LangChain

This project is divided into 3 milestones:

**1. Prepare the Data and Split it using LangChain**
Estimated duration: 1-2 hours

**2. Create a Vector Store and Embed Documents**
Estimated duration: 1-2 hours

**3. Use LangChain to Combine an LLM and the Vector Store**
Estimated duration: 1-3 hours

The deliverable will be an app that ingests data and inserts it into a vector database to be consumed by an LLM for retrieval-augmented generation, all using LangChain.

## Workflow

**Objective**

Prepare search data by extracting it, parsing it, and splitting it into chunks using LangChain.

**Importance to project**

Vector databases are essential in extending the data reach of LLMs, though these models’ limited context windows require data to be broken down for efficient processing and storage. Within this dynamic, LangChain provides a simplified framework for developing applications powered by LLMs, aiding in data integration and supporting various use cases from document analysis to chatbot creation.

**Workflow**

Step 1: Load documents

- Use LangChain’s ReadTheDocs loader to fetch documents from the specified source.
- Count the total number of documents loaded to check if it worked.

Step 2: Tokenization with tiktoken

- Leverage the tiktoken library for tokenizing text.
- Define a function to compute and return the length of tokens in a text.

Step 3: Splitting the text

- Employ LangChain’s `RecursiveCharacterTextSplitter` to divide large texts into manageable chunks.
- Set parameters like desired chunk size, overlap between chunks, and splitting criteria.

Step 4: Generate document chunks with unique IDs

- Iteratively process each document, extracting its URL and generating a unique ID.
- Split the document content into chunks.
- For each chunk, create a dictionary with its unique ID, text content, and source URL.
- Calculate the final number of chunks produced.

Step 5:

- Save the documents into a .jsonl file locally for easier access later.

**Deliverable**

The deliverable for this milestone is a Python program with the following functions:

- A function that reads the HTML documents.
- A function that computes and returns the number of tokens for a piece of text.
- A function that splits text into chunks.
- A function that generates one dictionary per chunk with its corresponding text, source, and ID.
- A function that converts that list of dictionaries into a .jsonl file.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

## 2.2 **Create a Vector Store and Embed Documents**

**Objective**

Set up a vector database using ChromaDB.

**Importance to project**

Embeddings significantly enhance LLMs’ ability to understand and analyze data, enabling nuanced semantic searches and idea linkage. LangChain simplifies building applications with LLMs, and ChromaDB, a specialized user-friendly platform for storing embeddings, optimizes this process and further enriches AI application development with cloud capabilities.

**Workflow**

Step 1: Setup and preparation

- Install necessary libraries from the [Setup section](https://liveproject.manning.com/module/1644_2_1).
- Set up the API key and import essential modules.

Step 2: Embedding and storage initialization

- Define the model and establish an embedding function.
- Set up a vector storage system using ChromaDB.

Step 3: Vectorize and store documents

- Iterate through the documents. This process might take up to 20 minutes depending on your computer specifications.
- Add each document’s vector representation to the storage system.

**Deliverable**

The deliverable for this milestone is python file that includes:

- A function to create embeddings using OpenAI.
- An instance of ChromaDB.
- A function the inserts documents into the vector store.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

## 2.3 Use LangChain to Combine an LLM and the Vector Store 

**Objective**

Implement a LangChain chain that uses the data from the vector store to give context to the LLM and answer questions.

**Importance to project**

Once we have the embeddings, we can use them to augment the knowledge of the LLM by inserting them into the prompt. This will radically improve the quality of the answers.

**Workflow**

Step 1: Initialize a retriever

- Use the vector storage system you created in Milestone 2 to initialize a retriever.

Step 2: Set up the chat model and Retrieval QA

- Import necessary modules.
- Create an LLM instance that uses the retriever.
- Be careful to define the chain type that best suits your purposes.

Step 3: Run queries

- Execute the query using the new LLM with retrieval.

**Deliverable**

The deliverable for this milestone is a Python file with the following:

- An instance of your vector store initialized as a retriever object.
- An LLM instance that uses the retriever object to get additional context and takes a query as an argument, returning the appropriate content.

Upload a link to your deliverable in the Submit Your Work section and click submit. After submitting, the author’s solution and peer solutions will appear on the page for you to examine.

**Additional resources**

LangChain’s documentation will provide you with an explanation and examples of how to implement one of their wrappers specialized for question-and-answer LLMs:

- [Using a Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Chain Type](https://python.langchain.com/docs/modules/chains/#legacy-chains)

If you want to improve the performance of your vector database, you might want to consider using an [index](https://python.langchain.com/docs/modules/data_connection/indexing) or adding metadata to your embeddings to get more information out of them.

## Project Conclusions

Another important step in enhancing Large Language Models (LLMs) through document retrieval techniques using the LangChain framework. The LangChain framework, with its multiple integrations and orchestration features, represents a significant stride in the advancement of Large Language Models (LLMs). By tapping into LangChain’s capabilities, users can seamlessly bridge LLMs with real-time data retrieval to offer deeper, more nuanced AI-driven interactions. The inclusion of vector databases ensures efficient handling of high-dimensional data, facilitating rapid responses to intricate queries.

# Part 3: Build and deploy the 

## about this liveProject

In the evolving digital landscape, enterprises are grappling with the challenge of tapping into extensive and dynamic data repositories. The proliferation of online platforms has underscored the urgent need for systems that can facilitate swift, precise data retrieval. For software engineers like you who are keen on exploring the potential of large language models, this presents an intriguing opportunity.

InfoHub stands at the forefront of this challenge. With a clear objective, this startup is charting a path to reshape knowledge-base interactions. They aim to engineer a system in which varied user queries, from technical support questions to searches for product specifications, find their answers in tailored datasets. Central to InfoHub’s blueprint is leveraging cutting-edge language models to create intuitive and agile question-and-answer interfaces customized for each organization’s data ecosystem.

As you embark on this project with LangServe, a key feature of LangChain, you’ll acquire the tools and knowledge to meet these evolving challenges. LangServe facilitates the deployment of sturdy APIs for models, operationalizing your LLMs—a pivotal step to make InfoHub both accessible and resilient. You’ll venture into creating a LangChain server application with OpenAI embeddings and the Chroma vector store. As you master the creation of conversational retrieval chains, you’ll use LangChain’s refined interfaces to infuse human-like dialog capabilities into your applications. The synergy between LangChain and Streamlit will guide you in crafting an interactive chatbot to improve user engagement and ensure global, user-friendly access to InfoHub’s data.

## Techniques employed

- LangServe: a tool to easily deploy LangChain apps with an API to access them.
- Streamlit: an open-source framework to build frontend applications for machine learning.
- The LangChain Framework: A platform that simplifies the integration of large language models into applications like document analysis and chatbot creation.
- Chroma Vector Store: Enables efficient storage and management of high-dimensional data such as embeddings and facilitates fast data retrieval.

## Project outline

This project is divided into 3 milestones:

**1. Deploy the Langserve Server**
Estimated duration: 1-2 hours

**2. Build a Chatbot using Streamlit**
Estimated duration: 1-2 hours

**3. Deploy to Streamlit Cloud**
Estimated duration: 1-2 hours

The deliverable for this project is a chatbot app that answers questions related to the LangChain documentation using LLMs, hosted on Streamlit Cloud.

## Deploy the LangServe Server

**Introduction**



[Animated Intro (0:25)](https://liveproject.manning.com/module/1645_1_1)



[About this liveProject](https://liveproject.manning.com/module/1645_1_2)



[Prerequisites and Resources](https://liveproject.manning.com/module/1645_1_4)



[How to Use Help](https://liveproject.manning.com/module/1645_1_5)



[How to Submit Your Work](https://liveproject.manning.com/module/1645_1_6)



[Are You Ready?](https://liveproject.manning.com/module/1645_1_7)

**Setup**

**1. Deploy the LangServe Server**



1.1 Workflow: Deploy the LangServe Server



[1.2 Submit Your Work](https://liveproject.manning.com/module/1645_3_2)

**2. Build a Chatbot using Streamlit**



[2.1 Workflow: Build a Chatbot using Streamlit](https://liveproject.manning.com/module/1645_4_1)



[2.2 Submit Your Work](https://liveproject.manning.com/module/1645_4_2)

**3. Deploy to the Streamlit Cloud**



[3.1 Workflow: Deploy to the Streamlit Cloud](https://liveproject.manning.com/module/1645_5_1)



[3.2 Submit Your Work](https://liveproject.manning.com/module/1645_5_2)

**Summary**

**FAQs**



[FAQs](https://liveproject.manning.com/module/1645_7_1)

**Series Navigator**



# Deploy the LangServe Server

Important! Be sure to read [About this liveProject](https://liveproject.manning.com/module/1645_1_3) before beginning. It contains crucial information for your work.

**Objective**

- Deploy your LLM application using LangServe

**Importance to project**

- In the evolving landscape of LLMs, LangChain’s LangServe stands out by providing a streamlined pathway for users to engage with LLM applications through easily accessible API endpoints. This integration fosters innovative applications and operational efficiency. Complementing this, the advent of chatbots as user interfaces has transformed interactions with LLMs, and Streamlit’s robust cloud platform is a leading choice for building and hosting these sophisticated systems. This dual advancement marks a significant stride in making LLM technology accessible, blending user-centric design with the power of cloud computing to deliver real-time, interactive applications that will drive the next phase of digital communication and data management.

**Introduction**



[Animated Intro (0:25)](https://liveproject.manning.com/module/1645_1_1)



[About this liveProject](https://liveproject.manning.com/module/1645_1_2)



[Prerequisites and Resources](https://liveproject.manning.com/module/1645_1_4)



[How to Use Help](https://liveproject.manning.com/module/1645_1_5)



[How to Submit Your Work](https://liveproject.manning.com/module/1645_1_6)



[Are You Ready?](https://liveproject.manning.com/module/1645_1_7)

**Setup**

**1. Deploy the LangServe Server**



1.1 Workflow: Deploy the LangServe Server



[1.2 Submit Your Work](https://liveproject.manning.com/module/1645_3_2)

**2. Build a Chatbot using Streamlit**



[2.1 Workflow: Build a Chatbot using Streamlit](https://liveproject.manning.com/module/1645_4_1)



[2.2 Submit Your Work](https://liveproject.manning.com/module/1645_4_2)

**3. Deploy to the Streamlit Cloud**



[3.1 Workflow: Deploy to the Streamlit Cloud](https://liveproject.manning.com/module/1645_5_1)



[3.2 Submit Your Work](https://liveproject.manning.com/module/1645_5_2)

**Summary**

**FAQs**



[FAQs](https://liveproject.manning.com/module/1645_7_1)

**Series Navigator**



# Deploy the LangServe Server

Important! Be sure to read [About this liveProject](https://liveproject.manning.com/module/1645_1_3) before beginning. It contains crucial information for your work.

**Objective**

- Deploy your LLM application using LangServe.

**Importance to project**

- In the evolving landscape of LLMs, LangChain’s LangServe stands out by providing a streamlined pathway for users to engage with LLM applications through easily accessible API endpoints. This integration fosters innovative applications and operational efficiency. Complementing this, the advent of chatbots as user interfaces has transformed interactions with LLMs, and Streamlit’s robust cloud platform is a leading choice for building and hosting these sophisticated systems. This dual advancement marks a significant stride in making LLM technology accessible, blending user-centric design with the power of cloud computing to deliver real-time, interactive applications that will drive the next phase of digital communication and data management.

**Workflow**

Step 1:

Set up the environment, retrieve API keys, and specify the model for embeddings.

Step 2:

Establish an embedding object with OpenAI and prepare the Chroma vector store instance. Set up the vector store as a retriever using LangChain.

Step 3:

Organize the LangChain components, including a memory buffer to store chat histories and a conversational retrieval chain. Test with an initial message for verification.

**Introduction**



[Animated Intro (0:25)](https://liveproject.manning.com/module/1645_1_1)



[About this liveProject](https://liveproject.manning.com/module/1645_1_2)



[Prerequisites and Resources](https://liveproject.manning.com/module/1645_1_4)



[How to Use Help](https://liveproject.manning.com/module/1645_1_5)



[How to Submit Your Work](https://liveproject.manning.com/module/1645_1_6)



[Are You Ready?](https://liveproject.manning.com/module/1645_1_7)

**Setup**

**1. Deploy the LangServe Server**



1.1 Workflow: Deploy the LangServe Server



[1.2 Submit Your Work](https://liveproject.manning.com/module/1645_3_2)

**2. Build a Chatbot using Streamlit**



[2.1 Workflow: Build a Chatbot using Streamlit](https://liveproject.manning.com/module/1645_4_1)



[2.2 Submit Your Work](https://liveproject.manning.com/module/1645_4_2)

**3. Deploy to the Streamlit Cloud**



[3.1 Workflow: Deploy to the Streamlit Cloud](https://liveproject.manning.com/module/1645_5_1)



[3.2 Submit Your Work](https://liveproject.manning.com/module/1645_5_2)

**Summary**

**FAQs**



[FAQs](https://liveproject.manning.com/module/1645_7_1)

**Series Navigator**



# Deploy the LangServe Server

Important! Be sure to read [About this liveProject](https://liveproject.manning.com/module/1645_1_3) before beginning. It contains crucial information for your work.

**Objective**

- Deploy your LLM application using LangServe.

**Importance to project**

- In the evolving landscape of LLMs, LangChain’s LangServe stands out by providing a streamlined pathway for users to engage with LLM applications through easily accessible API endpoints. This integration fosters innovative applications and operational efficiency. Complementing this, the advent of chatbots as user interfaces has transformed interactions with LLMs, and Streamlit’s robust cloud platform is a leading choice for building and hosting these sophisticated systems. This dual advancement marks a significant stride in making LLM technology accessible, blending user-centric design with the power of cloud computing to deliver real-time, interactive applications that will drive the next phase of digital communication and data management.

**Workflow**

Step 1:

Set up the environment, retrieve API keys, and specify the model for embeddings.

Step 2:

Establish an embedding object with OpenAI and prepare the Chroma vector store instance. Set up the vector store as a retriever using LangChain.

Step 3:

Organize the LangChain components, including a memory buffer to store chat histories and a conversational retrieval chain. Test with an initial message for verification.

help

Step 4:

Configure the FastAPI server, use LangServe to add the necessary API routes for interaction, and prepare the server for deployment.

Step 5:

Deploy the server with Uvicorn, making the application live for handling conversations and processing requests.

Run the Python file and make sure it’s working by going to localhost:8000/docs. You should see the API documentation.

**Introduction**



[Animated Intro (0:25)](https://liveproject.manning.com/module/1645_1_1)



[About this liveProject](https://liveproject.manning.com/module/1645_1_2)



[Prerequisites and Resources](https://liveproject.manning.com/module/1645_1_4)



[How to Use Help](https://liveproject.manning.com/module/1645_1_5)



[How to Submit Your Work](https://liveproject.manning.com/module/1645_1_6)



[Are You Ready?](https://liveproject.manning.com/module/1645_1_7)

**Setup**

**1. Deploy the LangServe Server**



1.1 Workflow: Deploy the LangServe Server



[1.2 Submit Your Work](https://liveproject.manning.com/module/1645_3_2)

**2. Build a Chatbot using Streamlit**



[2.1 Workflow: Build a Chatbot using Streamlit](https://liveproject.manning.com/module/1645_4_1)



[2.2 Submit Your Work](https://liveproject.manning.com/module/1645_4_2)

**3. Deploy to the Streamlit Cloud**



[3.1 Workflow: Deploy to the Streamlit Cloud](https://liveproject.manning.com/module/1645_5_1)



[3.2 Submit Your Work](https://liveproject.manning.com/module/1645_5_2)

**Summary**

**FAQs**



[FAQs](https://liveproject.manning.com/module/1645_7_1)

**Series Navigator**



# Deploy the LangServe Server

Important! Be sure to read [About this liveProject](https://liveproject.manning.com/module/1645_1_3) before beginning. It contains crucial information for your work.

**Objective**

- Deploy your LLM application using LangServe.

**Importance to project**

- In the evolving landscape of LLMs, LangChain’s LangServe stands out by providing a streamlined pathway for users to engage with LLM applications through easily accessible API endpoints. This integration fosters innovative applications and operational efficiency. Complementing this, the advent of chatbots as user interfaces has transformed interactions with LLMs, and Streamlit’s robust cloud platform is a leading choice for building and hosting these sophisticated systems. This dual advancement marks a significant stride in making LLM technology accessible, blending user-centric design with the power of cloud computing to deliver real-time, interactive applications that will drive the next phase of digital communication and data management.

**Workflow**

Step 1:

Set up the environment, retrieve API keys, and specify the model for embeddings.

Step 2:

Establish an embedding object with OpenAI and prepare the Chroma vector store instance. Set up the vector store as a retriever using LangChain.

Step 3:

Organize the LangChain components, including a memory buffer to store chat histories and a conversational retrieval chain. Test with an initial message for verification.

help

Step 4:

Configure the FastAPI server, use LangServe to add the necessary API routes for interaction, and prepare the server for deployment.

Step 5:

Deploy the server with Uvicorn, making the application live for handling conversations and processing requests.

Run the Python file and make sure it’s working by going to localhost:8000/docs. You should see the API documentation.

***Author insights\***

Why LangServe?

At the MVP (Minimum Viable Product) stage, LangServe provides a simple way of deploying your model with API access. Once you move to production, you might want to consider more sophisticated frameworks like Django.

