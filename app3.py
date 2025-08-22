import streamlit as st
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from typing import List, Dict
import re
import os
from PIL import Image
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏊‍♂️ Swimming Coach AI",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body {
        background: linear-gradient(180deg, #e0f7fa 0%, #80deea 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        background: url('https://www.transparenttextures.com/patterns/wavecut.png'), linear-gradient(135deg, #0288d1 0%, #0277bd 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 20px;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 100"><path fill="rgba(255,255,255,0.3)" d="M0,64L48,58.7C96,53,192,43,288,48C384,53,480,75,576,80C672,85,768,75,864,69.3C960,64,1056,64,1152,58.7C1248,53,1344,43,1392,37.3L1440,32L1440,100L1392,100C1344,100,1248,100,1152,100C1056,100,960,100,864,100C768,100,672,100,576,100C480,100,384,100,288,100C192,100,96,100,48,100L0,100Z"></path></svg>');
        animation: wave 5s linear infinite;
    }
    @keyframes wave {
        0% { background-position-x: 0; }
        100% { background-position-x: 1440px; }
    }
    .main-header h1 {
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-in;
    }
    .main-header p, .main-header small {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .chat-message:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    .user-message {
        background: linear-gradient(135deg, #4fc3f7 0%, #0288d1 100%);
        color: white;
        margin-left: 3rem;
    }
    .assistant-message {
        background: linear-gradient(135deg, #26a69a 0%, #00796b 100%);
        color: white;
        margin-right: 3rem;
    }
    .stTextArea textarea {
        border: 2px solid #4fc3f7;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #0288d1;
        box-shadow: 0 0 8px rgba(2, 136, 209, 0.5);
    }
    .stButton button {
        background: linear-gradient(135deg, #0288d1 0%, #4fc3f7 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 1.5rem;
        font-weight: bold;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .image-container img {
        border-radius: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .image-container img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .source-item {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #4fc3f7;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        transition: background 0.3s ease;
    }
    .source-item:hover {
        background: rgba(255, 255, 255, 1);
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        background: linear-gradient(135deg, #0288d1 0%, #0277bd 100%);
        color: white;
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        position: relative;
        overflow: hidden;
    }
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 20px;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 100"><path fill="rgba(255,255,255,0.3)" d="M0,64L48,58.7C96,53,192,43,288,48C384,53,480,75,576,80C672,85,768,75,864,69.3C960,64,1056,64,1152,58.7C1248,53,1344,43,1392,37.3L1440,32L1440,100L1392,100C1344,100,1248,100,1152,100C1056,100,960,100,864,100C768,100,672,100,576,100C480,100,384,100,288,100C192,100,96,100,48,100L0,100Z"></path></svg>');
        animation: wave 5s linear infinite reverse;
    }
</style>
""", unsafe_allow_html=True)

class SwimmingAssistant:
    def __init__(self):
        self.setup_session_state()
        self.search_model, self.embeddings, self.chunks, self.index, self.image_index, self.image_chunks = self.load_models_and_data()
    
    def setup_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    @st.cache_resource
    def load_models_and_data(_self):
        try:
            with st.spinner("Loading embedding model..."):
                search_model = SentenceTransformer('all-mpnet-base-v2')
            
            base_paths = ["output1", "output", "data"]
            embeddings, chunks = None, None
            
            for base_path in base_paths:
                embeddings_path = f"{base_path}/embeddings_swimming.pkl"
                chunks_path = f"{base_path}/chunks_prepared.json"
                
                if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
                    with st.spinner("Loading embeddings..."):
                        with open(embeddings_path, 'rb') as f:
                            data = pickle.load(f)
                            embeddings = data['embeddings']
                    
                    with st.spinner("Loading chunks..."):
                        with open(chunks_path, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                    break
            
            if embeddings is None or chunks is None:
                st.warning("Training data not found. Basic functionality available.")
                return search_model, None, [], None, None, []
            
            if len(chunks) != embeddings.shape[0]:
                min_size = min(len(chunks), embeddings.shape[0])
                chunks = chunks[:min_size]
                embeddings = embeddings[:min_size]
            
            with st.spinner("Building FAISS index..."):
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings.astype('float32'))
            
            image_chunks = [c for c in chunks if c.get('images')]
            if image_chunks:
                image_embeddings = np.array([embeddings[i] for i, c in enumerate(chunks) if c.get('images')])
                image_index = faiss.IndexFlatIP(dimension)
                image_index.add(image_embeddings.astype('float32'))
            else:
                image_index = None
                image_chunks = []
            
            return search_model, embeddings, chunks, index, image_index, image_chunks
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, [], None, None, []
    
    def get_api_key(self):
        try:
            return st.secrets["GROK_API_KEY"]
        except:
            pass
        
        api_key = os.environ.get('GROK_API_KEY')
        return api_key if api_key else None
    
    def generate_with_grok(self, prompt: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an intelligent sports assistant specializing in swimming, with expertise in technical coaching and training strategies. Respond in a way that is:
- Precise and actionable (limit to 100-150 words).
- Based on scientific data.
- Safe, prioritizing injury prevention.
- Encouraging and motivating."""
        
        if any(keyword in prompt.lower() for keyword in ["nutrition", "diet", "injury", "recovery"]):
            system_prompt += "\n- Include advice on nutrition or injury prevention as relevant to the query."
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return "API Error: Unable to process request."
                
        except Exception as e:
            return "Network Error: Unable to connect to API."
    
    def search_documents(self, query: str, top_k: int = 4) -> List[Dict]:
        if not hasattr(self, 'index') or self.index is None:
            return []
        
        try:
            enriched_query = f"{query} swimming technique"
            
            query_embedding = self.search_model.encode([enriched_query], normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
            
            results = []
            seen_content = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if score < 0.12 or idx >= len(self.chunks):
                    continue
                
                chunk = self.chunks[idx].copy()
                content_hash = hash(chunk['text'][:150])
                
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                chunk['similarity_score'] = float(score)
                chunk['cleaned_text'] = self.clean_text_for_llm(chunk['text'])
                
                if len(chunk['cleaned_text']) > 30:
                    results.append(chunk)
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def search_image_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        if not hasattr(self, 'image_index') or self.image_index is None:
            return []
        
        try:
            enriched_query = f"{query} swimming technique"
            query_embedding = self.search_model.encode([enriched_query], normalize_embeddings=True)
            scores, indices = self.image_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score < 0.12 or idx >= len(self.image_chunks):
                    continue
                chunk = self.image_chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
            
            return results
            
        except Exception as e:
            st.error(f"Image search error: {str(e)}")
            return []
    
    def clean_text_for_llm(self, text: str) -> str:
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'About the book.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ISBN.*?(?=\n|$)', '', text)
        text = re.sub(r'www\..*?(?=\s|$)', '', text)
        text = re.sub(r'\$\s*\d+\.\d+.*?(?=\s|$)', '', text)
        return text.strip()[:600]
    
    def extract_images(self, search_results: List[Dict], base_path: str = "output1/images_cleaned") -> List[Dict]:
        images = []
        if not os.path.exists(base_path):
            st.error(f"Image directory not found: {base_path}")
            return images
        
        for result in search_results:
            if result.get('images'):
                for img in result['images'][:2]:
                    relative_path = img.get('path', '')
                    if relative_path:
                        if relative_path.startswith('images_cleaned/'):
                            relative_path = relative_path[len('images_cleaned/'):]
                        full_path = os.path.join(base_path, relative_path)
                        full_path = os.path.normpath(full_path)
                        if os.path.exists(full_path):
                            images.append({
                                'path': full_path,
                                'source': result['pdf'].replace('.pdf', ''),
                                'page': result['page'],
                                'description': img.get('description', 'Swimming technique illustration')
                            })
                        else:
                            st.warning(f"Image file not found: {full_path}")
        
        if not images and hasattr(self, 'image_index') and self.image_index:
            fallback_results = self.search_image_chunks(search_results[0]['cleaned_text'] if search_results else '')
            for result in fallback_results:
                if result.get('images'):
                    for img in result['images'][:2]:
                        relative_path = img.get('path', '')
                        if relative_path:
                            if relative_path.startswith('images_cleaned/'):
                                relative_path = relative_path[len('images_cleaned/'):]
                            full_path = os.path.join(base_path, relative_path)
                            full_path = os.path.normpath(full_path)
                            if os.path.exists(full_path):
                                images.append({
                                    'path': full_path,
                                    'source': result['pdf'].replace('.pdf', ''),
                                    'page': result['page'],
                                    'description': img.get('description', 'Swimming technique illustration')
                                })
                            else:
                                st.warning(f"Image file not found: {full_path}")
        
        return images[:3] if images else [{'path': 'output1/images_cleaned/wave_placeholder.png', 'source': 'Default', 'page': 0, 'description': 'Swimming illustration'}]
    
    def display_image_in_streamlit(self, image_path: str) -> bool:
        try:
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            with open(image_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode()
            st.markdown(f'<div class="image-container"><img src="data:{mime_type};base64,{img_b64}" width="100%" alt="Swimming technique illustration"></div>', unsafe_allow_html=True)
            return True
        except Exception as e:
            st.error(f"Failed to display image {image_path}: {str(e)}")
            return False
    
    def build_enhanced_prompt(self, query: str, search_results: List[Dict]) -> str:
        context_parts = []
        
        if search_results:
            doc_context = []
            for i, result in enumerate(search_results[:3], 1):
                source = result['pdf'].replace('.pdf', '').replace('_', ' ').title()
                text = result['cleaned_text']
                doc_context.append(f"Source {i} ({source}): {text}")
            context_parts.append("TECHNICAL DOCUMENTATION:\n" + '\n\n'.join(doc_context))
        
        full_context = '\n\n'.join(context_parts)
        
        return f"""{full_context}

QUESTION: {query}

As an intelligent swimming assistant, provide a concise response that:
1. Offers precise and safe technical advice
2. Suggests concrete, progressive actions
3. Motivates and encourages the user

Response:"""
    
    def is_query_valid(self, query: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> tuple[bool, str]:
        query_clean = query.strip()
        if not query_clean:
            return False, "Please enter a valid swimming-related question, e.g., 'How to improve my freestyle technique?'"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an intelligent validator for a swimming coach AI. Your task is to determine if a query is:
1. Relevant to swimming , sport , training 
2. Not offensive or inappropriate.
3. Comprehensible (not random characters or gibberish).

Return one word: 'valid', 'off-topic', 'offensive', or 'gibberish'."""
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_clean}
            ],
            "max_tokens": 10,
            "temperature": 0.5
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                validation = result['choices'][0]['message']['content'].strip().lower()
                if validation == 'valid':
                    return True, ""
                elif validation == 'offensive':
                    return False, "Please use appropriate language and ask a swimming-related question."
                else:
                    return False, "Please enter a valid swimming-related question, e.g., 'How to improve my freestyle technique?'"
            else:
                return False, "API Error: Unable to validate query."
                
        except Exception as e:
            return False, "Network Error: Unable to validate query."
    
    def validate_query(self, query: str, api_key: str) -> tuple[bool, str]:
        return self.is_query_valid(query, api_key)
    
    def process_enhanced_query(self, query: str, model_name: str = "llama-3.3-70b-versatile") -> Dict:
        api_key = self.get_api_key()
        if not api_key:
            return {
                'text': "Grok API key required. Please set GROK_API_KEY in environment variables or Streamlit secrets.",
                'images': [{'path': 'output1/images_cleaned/wave_placeholder.png', 'source': 'Default', 'page': 0, 'description': 'Swimming illustration'}],
                'sources': []
            }
        
        is_valid, error_message = self.validate_query(query, api_key)
        if not is_valid:
            return {
                'text': error_message,
                'images': [{'path': 'output1/images_cleaned/wave_placeholder.png', 'source': 'Default', 'page': 0, 'description': 'Swimming illustration'}],
                'sources': []
            }
        
        search_results = self.search_documents(query)
        prompt = self.build_enhanced_prompt(query, search_results)
        llm_response = self.generate_with_grok(prompt, api_key, model_name)
        images = self.extract_images(search_results)
        
        return {
            'text': llm_response,
            'images': images,
            'sources': [f"{r['pdf'].replace('.pdf', '').replace('_', ' ').title()} (p.{r['page']})" 
                       for r in search_results]
        }

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🏊‍♂️ Swimming Coach AI</h1>
        <p>Dive into personalized swimming coaching powered by AI</p>
        <small>Master your technique • Swim faster • Achieve greatness</small>
    </div>
    """, unsafe_allow_html=True)
    
    if 'swimming_assistant' not in st.session_state:
        with st.spinner("Initializing swimming coach..."):
            st.session_state.swimming_assistant = SwimmingAssistant()
    
    assistant = st.session_state.swimming_assistant
    
    st.markdown("### 💬 Dive into Your Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area(
            "",
            height=100,
            placeholder="Ex: How to improve my freestyle swimming technique?"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            ask_button = st.button("Ask Coach", type="primary")
        with col_btn2:
            clear_button = st.button("Clear History")
    
    with col2:
        st.markdown("### 💡 Swim Smart: Try These Questions")
        suggested_questions = [
            "How to improve my freestyle technique?",
            "Best training plan for competitive swimming?",
            "Tips for better butterfly stroke?",
            "How to increase swimming endurance?",
            "Drills for improving backstroke?"
        ]
        
        for i, suggestion in enumerate(suggested_questions):
            if st.button(f"🏊‍♂️ {suggestion}", key=f"suggestion_{i}"):
                user_query = suggestion
                ask_button = True
    
    if ask_button and user_query.strip():
        with st.spinner("Analyzing your question..."):
            response = assistant.process_enhanced_query(user_query)
            st.session_state.chat_history.append({
                'question': user_query,
                'response': response,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
    
    if clear_button:
        st.session_state.chat_history = []
        st.success("History cleared!")
    
    if st.session_state.chat_history:
        st.markdown("### Your Swim Journey")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🏊‍♀️ You ({chat['timestamp']}):</strong><br>
                    {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🏊‍♂️ Swim Coach:</strong><br>
                    {chat['response']['text']}
                </div>
                """, unsafe_allow_html=True)
                
                if chat['response'].get('sources'):
                    with st.expander("Sources Consulted"):
                        for j, source in enumerate(chat['response']['sources'], 1):
                            st.markdown(f"""
                            <div class="source-item">
                                {j}. {source}
                            </div>
                            """, unsafe_allow_html=True)
                
                if chat['response'].get('images'):
                    with st.expander(f"Visual Swim Guides ({len(chat['response']['images'])})"):
                        cols = st.columns(min(len(chat['response']['images']), 3))
                        for idx, img in enumerate(chat['response']['images']):
                            with cols[idx % 3]:
                                if assistant.display_image_in_streamlit(img['path']):
                                    st.caption(f"{img['source']} - Page {img['page']}")
                
                st.divider()

    st.markdown("""
    <div class="footer">
        🏊‍♂️ <strong>Swimming Coach AI</strong> | 
        Powered by Grok AI & RAG Technology<br>
        <small>Ride the wave to swimming excellence</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()