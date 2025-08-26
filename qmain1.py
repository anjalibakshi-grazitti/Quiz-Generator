#HireIT api key quiz generator
#Quiz can be generated either by topic or a document. No of questions can be changed by changing the prompt
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import json
import re
import base64
import requests

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type!")
    return loader.load()



def generate_quiz(content, difficulty):
    prompt = f"""
You are an AI tutor who generates educational content. Create a quiz based on the following content:
Strictly return only a valid JSON array of 20 multiple-choice questions (MCQs). Do not include any explanation, notes, or introductory text—only the pure JSON list.
{content}

Generate 20 multiple-choice questions (MCQs) with the following specifications:
- Difficulty level: {difficulty}
- Each question should have 4 options (A, B, C, D)
- Include the correct answer
- Format each question as a JSON object

Return the output as a list of JSON objects in the following format:
[
    {{
        "skills": "topic, subtopic",
        "difficulty": "{difficulty.lower()}",
        "name": "question text",
        "score": "1",
        "options": ["option A", "option B", "option C", "option D"],
        "correctOption": "correct option text",
        "type": "MCQ"
    }},
    // ... more questions ...
]

Make sure:
1. Questions are clear and unambiguous
2. Options are well-distributed and logical
3. Correct answers are accurate
4. JSON format is strictly followed
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using the correct model
        messages=[
            {"role": "system", "content": "You are a helpful educational assistant that generates well-structured quizzes in JSON format. Adjust the level of quiz complexity to the '{difficulty}' level."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Adding temperature for better creativity
        max_tokens=2000   # Ensuring enough tokens for complete response
    )
    return response.choices[0].message.content

def create_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 200 )
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(splits, embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever

def extract_json_array(text):
    """
    Extracts the first JSON array found in a string, even if preceded by 'json' or markdown code block.
    Returns the JSON array as a string, or None if not found.
    """
    # Remove markdown code block and 'json' prefix if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("json"):
        text = text[len("json"):].strip()
    # Now extract the array
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        return match.group(1)
    return None


st.set_page_config(page_title="Quiz Generator", layout="centered")
st.title("Objective Test Generator")


# Ensure session state key is always initialized
if 'last_generated_quiz' not in st.session_state:
    st.session_state.last_generated_quiz = None

input_method = st.radio(
    "Choose input method:",
    ["Topic-based (enter a topic)", "Document-based (upload a document)"]
)

difficulty = st.selectbox("Select quiz difficulty:", ["Easy", "Medium", "Hard"])

if input_method == "Topic-based (enter a topic)":
    topic_input = st.text_input("Enter a topic:", placeholder="e.g., Artificial Intelligence")
    if st.button("Generate Test"):
        if topic_input.strip() == "":
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating test..."):
                result = generate_quiz(topic_input, difficulty)
                if not result.strip():
                    st.error("No response from the model. Please try again.")
                else:
                    json_str = result
                    try:
                        quiz_data = json.loads(json_str)
                        st.session_state.last_generated_quiz = quiz_data
                        st.success("Quiz Generated Successfully!")
                    except json.JSONDecodeError:
                        # Try to extract JSON array from the text
                        json_array = extract_json_array(result)
                        if json_array:
                            try:
                                quiz_data = json.loads(json_array)
                                st.session_state.last_generated_quiz = quiz_data
                                st.success("Quiz Generated Successfully (after extraction)!")
                            except Exception:
                                st.error("Could not parse extracted JSON array. Please check the model output.")
                                st.write("Extracted JSON:", json_array)
                        else:
                            st.error("The model did not return valid JSON. Please try again or check your prompt.")
                            st.write("Raw response:", result)

    # Always display from session_state
    if st.session_state.last_generated_quiz:
        for i, question in enumerate(st.session_state.last_generated_quiz, 1):
            st.markdown(f"### Question {i}")
            st.write(question["name"])
            options = question["options"]
            for j, option in enumerate(options):
                st.write(f"{chr(65+j)}. {option}")
            with st.expander("Show Answer"):
                st.write(f"Correct Answer: {question['correctOption']}")
            st.markdown("---")

elif input_method == "Document-based (upload a document)":
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, DOCX):", type=["pdf", "txt", "docx"])
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save file temporarily
            file_path = os.path.join("temp_" + uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                docs = load_document(file_path)
                doc_text = "\n".join([doc.page_content for doc in docs])

                with st.spinner("Generating quiz..."):
                    result = ""
                    try:
                        result = generate_quiz(doc_text, difficulty)
                        # Display quiz in a more organized way
                        st.success("Quiz Generated Successfully!")
                        
                        for i, question in enumerate(json.loads(result), 1):
                            st.markdown(f"### Question {i}")
                            st.write(question["name"])
                            
                            # Display options
                            options = question["options"]
                            for j, option in enumerate(options):
                                st.write(f"{chr(65+j)}. {option}")
                            
                            # Show correct answer in expander
                            with st.expander("Show Answer"):
                                st.write(f"Correct Answer: {question['correctOption']}")
                            
                            st.markdown("---")
                    except json.JSONDecodeError:
                        # Try to extract JSON array from the text
                        json_array = extract_json_array(result)
                        if json_array:
                            try:
                                # Try to fix common JSON issues
                                fixed_json = json_array.replace(",\n]", "\n]")
                                quiz_data = json.loads(fixed_json)
                                st.session_state.last_generated_quiz = quiz_data
                                st.success("Quiz Generated Successfully (after extraction)!")
                                for i, question in enumerate(quiz_data, 1):
                                    st.markdown(f"### Question {i}")
                                    st.write(question["name"])
                                    options = question["options"]
                                    for j, option in enumerate(options):
                                        st.write(f"{chr(65+j)}. {option}")
                                    with st.expander("Show Answer"):
                                        st.write(f"Correct Answer: {question['correctOption']}")
                                    st.markdown("---")
                            except Exception as e:
                                st.error("Could not parse extracted JSON array. Please check the model output.")
                                st.write("Extracted JSON:", json_array)
                                st.write("Raw response:", result)
                        else:
                            # Fallback: Try to extract using regex
                            pattern = r"\d+\. (.*?)\n\s*A\. (.*?)\n\s*B\. (.*?)\n\s*C\. (.*?)\n\s*D\. (.*?)\n\s*Answer: (.*?)\n"
                            matches = re.findall(pattern, result, re.DOTALL)
                            if matches:
                                st.success("Quiz Generated Successfully (using regex fallback)!")
                                quiz_data = []
                                for i, match in enumerate(matches, 1):
                                    question_json = {
                                        "skills": "",
                                        "difficulty": difficulty.lower(),
                                        "name": match[0].strip(),
                                        "score": "1",
                                        "options": [match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()],
                                        "correctOption": match[5].strip(),
                                        "type": "MCQ"
                                    }
                                    quiz_data.append(question_json)
                                    st.markdown(f"### Question {i}")
                                    st.write(question_json["name"])
                                    for j, option in enumerate(question_json["options"]):
                                        st.write(f"{chr(65+j)}. {option}")
                                    with st.expander("Show Answer"):
                                        st.write(f"Correct Answer: {question_json['correctOption']}")
                                    st.markdown("---")
                                st.session_state.last_generated_quiz = quiz_data
                            else:
                                # If all fails, show the raw output for debugging
                                st.error("Could not extract questions from the output. Please check the model output below.")
                                st.write("Raw response:", result)
            except Exception as e:
                st.error(f"Error displaying quiz: {str(e)}")
                st.write("Raw response:", result)
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

def upload_quiz(quiz_data): 
    url = "https://api-hireit.grazitti.com/question-mgmt/upload-json-questions"
    sec_api_key =  "2025-08-26:OSfJIyzeRBfp007zqcYD7KBf4"
    base64_api_key = base64.b64encode(sec_api_key.encode('utf-8')).decode('utf-8')

    headers = {
        "Authorization": f"Basic {base64_api_key}",
        "Content-Type": "application/json"
    } 

    try:
        response = requests.post(url, headers=headers, json=quiz_data)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Add this after the document-based upload section
st.markdown("---")
st.subheader("Upload Quiz to Grazitti")

# Add a button to upload the last generated quiz
if 'last_generated_quiz' not in st.session_state:
    st.session_state.last_generated_quiz = None

if st.session_state.last_generated_quiz:
    if st.button("Upload Quiz to Grazitti"):
        with st.spinner("Uploading quiz..."):
            upload_result = upload_quiz(st.session_state.last_generated_quiz)
            if "error" in upload_result:
                st.error(f"Upload failed: {upload_result['error']}")
            else:
                st.success("Quiz uploaded successfully!")
                st.json(upload_result)
else:
    st.info("Generate a quiz first to enable upload functionality")







