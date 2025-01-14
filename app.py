from twilio.rest import Client # type: ignore
import time
from pathlib import Path
# THis code is working fine for initial stage
from flask import Flask, request, jsonify
import model
import os
from dotenv import load_dotenv # type: ignore
from groq import Groq # type: ignore
import PyPDF2 # type: ignore
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import logging
from twilio.rest import Client # type: ignore
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
Account_SID = os.getenv("Account_SID")
Auth_Token = os.getenv("Auth_Token")
Twilio_Number= os.getenv("Twilio_Number")
Recipient_Number = os.getenv("Recipient_Number")


app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # response = {"reply": f"You said: {user_input}"}
    response= resp(user_input)
    return jsonify(response)


############################## whatsapp message send using TWilio ######################333
def send_to_whatsapp(conversation_log):
    try:
        # Twilio account credentials
        account_sid = Account_SID
        auth_token = Auth_Token
        client = Client(account_sid, auth_token)

        # Format the log as a readable string
        log_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_log])

        # Send to WhatsApp
        message = client.messages.create(
            from_=Twilio_Number,  # Twilio's sandbox WhatsApp number
            to= Recipient_Number,  # Your WhatsApp number
            body=f"Conversation Log:\n{log_text}"
        )
        print("WhatsApp message sent:", message.sid)
    except Exception as e:
        print("Failed to send WhatsApp message:", str(e))




# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print("Failed to initialize the AI model. Please check your API key.")


Model = "llama3-70b-8192"
# Model = "mixtral-8x7b-32768"

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_page = {executor.submit(extract_text, page): page for page in pdf_reader.pages}
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    text += future.result() + "\n"
                except Exception as e:
                    logger.warning(f"Skipped page {pdf_reader.pages.index(page)} due to: {str(e)}")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        print("Failed to process the PDF. Please try again with a different file.")
        return ""

def extract_text(page):
    try:
        return page.extract_text()
    except Exception as e:
        logger.warning(f"Error extracting text from page: {e}")
        return ""

def split_into_chunks(text, chunk_size=1000, overlap=100):

    logger.warning(f"Text for Cunks : {len(text)}")

    # Validate inputs
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0
    
    # Generate chunks with overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())  # Remove any trailing whitespace
        start += chunk_size - overlap  # Move the start pointer with overlap
    
    logger.warning(f"Chunks size is : {len(chunks)}")
    return chunks


def get_or_create_chunks(file_paths):
    try:
        combined_text = ""  # Initialize an empty string to hold combined text
        
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as file:
                    file_content = file.read()  # Read the file content as bytes
                    file_hash = hashlib.md5(file_content).hexdigest()

                cache_file = f"cache/{file_hash}_chunks.pkl"
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cached_chunks = pickle.load(f)
                        combined_text += " ".join(cached_chunks)  # Combine cached chunks
                        continue

                # If not cached, process the file and extract text
                with open(file_path, 'rb') as file:
                    try:
                        text = process_pdf(file)  # Replace with robust text extraction
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
                        continue

                    combined_text += text  # Combine text from all files
            except Exception as file_error:
                logger.warning(f"Skipping file {file_path} due to error: {file_error}")
                continue

        

        # Split the combined text into chunks
        chunks = split_into_chunks(combined_text)

        # Cache the chunks (use a single hash for the combined text)
        combined_hash = hashlib.md5(combined_text.encode('utf-8')).hexdigest()
        cache_file = f"cache/{combined_hash}_chunks.pkl"
        os.makedirs('cache', exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)

        return chunks
    except Exception as e:
        logger.error(f"Error in get_or_create_chunks: {e}")
        return []


def get_vectorizer(chunks):
    try:
        return TfidfVectorizer().fit(chunks)
    except Exception as e:
        logger.error(f"Error creating vectorizer: {e}")
        print("Failed to create text vectorizer. Please try again.")
        return None

def find_most_relevant_chunks(query, chunks, vectorizer,top_k):
    try:
        chunk_vectors = vectorizer.transform(chunks)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        print("Failed to find relevant information. Please try a different query.")
        return []

def get_ai_response(messages, context, model):
    try:
        system_message = {"role": "system", "content": "You are a helpful university chatbot assistant for answering university of sialkot related questions about the given PDF content. Use the provided context to answer questions, but also consider the conversation history. if the asnwer is not in context then say that bot is still under construction "}
        system_message2 = {"role": "system", "content": " This is the generate fee of programs and admisson, but it can be vary by speciic program so first take look in program detail if the details is note exsis then you can use it, Admission Fee: Rs. 15,000, Registration Fee: Rs. 15,000, Asosiative Degree Program (ADP) one semester fee is  86,515, Bs program one semester fee is 95,832, MS program one semester fee Rs 113,135, PHD program fee for one semester is Rs. 167,706 "}
        system_message3 = {"role": "system", "content": "write positive intro if someone ask about a personality that does not exist in context"}
        
        # Combine system message, conversation history, and the new query with context
        all_messages = [system_message]+ [system_message2]+ [system_message3]+ messages[:-1] + [{"role": "user", "content": f"Context: {context}\n\nBased on this context and our previous conversation, please answer the following question: {messages[-1]['content']}"}]

        chat_completion = client.chat.completions.create(
            messages=all_messages,
            model=model,
            max_tokens=1024,
            temperature=0.5
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return "I'm sorry, Chat session compelete please reset chat by clicking on reset button"

def resp(prompt):
    # Initialize session state variables using a dictionary
    session_state = {}
    if 'messages' not in session_state:
        session_state['messages'] = []

    if 'model' not in session_state:
        session_state['model'] = Model  # Replace with your actual Model object

    if 'chunks' not in session_state:
        session_state['chunks'] = []

    if 'vectorizer' not in session_state:
        session_state['vectorizer'] = None

    if 'conversation_log' not in session_state:
        session_state['conversation_log'] = []  # To store all messages in a session


    # pdf_file = ["./data/Directions_data.pdf", "./data/Fee Structure.pdf", "./data/General_data.pdf", "./data/Post-Graduate_Programs.pdf", "./data/Teachers data.pdf", "./data/Under_Graduate_Programs.pdf", "./data/University of Sialkot chatbot.pdf"]
    pdf_file = ["./data/Uskt_Data.pdf", "./data/Directions_data.pdf", "./data/ReTrain_Data.pdf"]

    if pdf_file:
        session_state['chunks'] = get_or_create_chunks(pdf_file)
        session_state['vectorizer'] = get_vectorizer(session_state['chunks'])
        if session_state['chunks'] and session_state['vectorizer']:
            print("PDF processed successfully!")
        else:
            print("Failed to process PDF. Please try again.")

    # Chat input
    if prompt:
        session_state['messages'].append({"role": "user", "content": prompt})
        session_state['conversation_log'].append({"role": "\n\n user", "content": prompt})
        
        relevant_chunks = find_most_relevant_chunks(prompt, session_state['chunks'], session_state['vectorizer'], top_k=6) if session_state['chunks'] else []
        context = "\n\n".join(relevant_chunks)

        # prompt_limit = f"{st.session_state.messages} + {context}"
        # render_message(f"Len of chunk0 = {len(relevant_chunks[0])} \n Len of chunk1 = {len(relevant_chunks[1])} \n Len of chunk2 =  total chunks length:{len(relevant_chunks[2])} \n  relevent chunk:{len(relevant_chunks)}---------prompt size: {len(prompt_limit)}--------session state message len: -{len(st.session_state.messages)}--------Context length: -{len(context)}", "assistant")
        full_response = get_ai_response(session_state['messages'], context, session_state['model'])

        session_state['messages'].append({"role": "assistant", "content": full_response})
        # st.session_state.conversation_log.append({"role": "assistant", "content": full_response})
        
        ################# Sending whatsapp log ##########################33
    
    # if len(st.session_state.conversation_log) >= 4:  # Example threshold
    #     # Send the log to WhatsApp
    #     send_to_whatsapp(st.session_state.conversation_log)
    #     # Optionally clear the log after sending
    #     print("whatsapp message send")
    #     st.session_state.conversation_log = []

    return full_response


if __name__ == '__main__':
    app.run(debug=True)
