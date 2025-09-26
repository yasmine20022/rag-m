"""
Advanced Swimming Coach AI - Production RAG with Enhanced Evaluation
Complete system implementing all priority recommendations
"""

import streamlit as st
import json
import pickle
import numpy as np
import faiss
import requests
from typing import List, Dict, Tuple, Optional, Any, Set, Union
import re
import os
from PIL import Image
import base64
from datetime import datetime, timedelta
import warnings
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import time
import hashlib
from functools import lru_cache
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pandas as pd
from collections import defaultdict, Counter
import uuid
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textstat import flesch_reading_ease
import sentence_transformers
from transformers import pipeline
from abc import ABC, abstractmethod



warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None

try:
    from textstat import flesch_reading_ease
except ImportError:
    def flesch_reading_ease(text):
        return 70  # Default readable score

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# ==================== ENHANCED CONFIGURATION ====================

@dataclass
class RAGConfig:
    """Enhanced configuration with new evaluation features"""
    # Core embeddings
    embedding_model_name: str = "all-mpnet-base-v2"
    embedding_dimension: int = 384
    normalize_embeddings: bool = True
    
    # Advanced retrieval
    top_k_retrieval: int = 10
    rerank_top_k: int = 15
    similarity_threshold: float = 0.01
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    enable_hybrid_search: bool = True
    
    # Generation
    max_tokens: int = 400
    temperature: float = 0.7
    max_retries: int = 3
    request_timeout: int = 35
    
    # Hallucination detection
    enable_hallucination_detection: bool = True
    hallucination_threshold: float = 0.3
    fact_check_threshold: float = 0.6
    
    # Reference-based evaluation
    reference_dataset_path: str = "reference_qa.json"
    enable_reference_evaluation: bool = True
    
    # User feedback
    enable_user_feedback: bool = True
    feedback_db_path: str = "user_feedback.db"
    
    # Degraded modes
    fallback_modes: Dict[str, Dict] = field(default_factory=lambda: {
        'basic': {'embedding_fallback': True, 'simple_search': True},
        'minimal': {'keyword_only': True, 'no_evaluation': True},
        'offline': {'static_responses': True, 'cached_only': True}
    })
    
    # Performance optimization
    cache_embeddings: bool = True
    cache_size: int = 1000
    batch_size: int = 32
    
    # Images
    max_images_per_response: int = 6
    max_image_size_mb: int = 5
    supported_formats: List[str] = field(default_factory=lambda: ['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
    
    # Evaluation thresholds
    evaluation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 80.0,
        'good': 65.0,
        'acceptable': 50.0,
        'poor': 35.0
    })
    
    # Data paths
    chunks_file: str = "chunks_prepared.json"
    embeddings_file: str = "embeddings_swimming.pkl"
    images_dirs: List[str] = field(default_factory=lambda: [
        "output1/images_cleaned,images_cleaned" 
    ])

class SystemMode(Enum):
    """Enhanced system modes"""
    FULL = "full"
    ADVANCED = "advanced"
    BASIC = "basic"
    MINIMAL = "minimal"
    OFFLINE = "offline"
    ERROR = "error"







class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer-based embedding provider"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.model:
            raise RuntimeError("Model not available")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model is not None

class FallbackEmbeddingProvider(EmbeddingProvider):
    """Fallback embedding provider using TF-IDF"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectorizer = TfidfVectorizer(max_features=dimension, stop_words='english')
        self.is_fitted = False
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF"""
        try:
            if not self.is_fitted:
                # Fit on the texts
                self.vectorizer.fit(texts)
                self.is_fitted = True
            
            tfidf_matrix = self.vectorizer.transform(texts)
            
            # Pad or truncate to desired dimension
            embeddings = tfidf_matrix.toarray()
            if embeddings.shape[1] < self.dimension:
                padding = np.zeros((embeddings.shape[0], self.dimension - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            elif embeddings.shape[1] > self.dimension:
                embeddings = embeddings[:, :self.dimension]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Fallback encoding error: {e}")
            # Return random embeddings as last resort
            return np.random.random((len(texts), self.dimension))
    
    def is_available(self) -> bool:
        """Always available"""
        return True

class EvaluationProvider(ABC):
    """Abstract base class for evaluation providers"""
    
    @abstractmethod
    def evaluate_retrieval(self, query: str, results: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate retrieval quality"""
        pass
    
    @abstractmethod
    def evaluate_generation(self, query: str, response: str, context: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate generation quality"""
        pass
    
    @abstractmethod
    def evaluate_multimodal(self, query: str, response: str, images: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate multimodal coherence"""
        pass

class FAISSSearchProvider:
    """FAISS-based search provider"""
    
    def __init__(self, embedding_provider: EmbeddingProvider, config: RAGConfig):
        self.embedding_provider = embedding_provider
        self.config = config
        self.chunks = []
        self.index = None
        self.embeddings = None
    
    def load_data(self, chunks: List[Dict], embeddings: Optional[np.ndarray] = None):
        """Load chunks and build search index"""
        try:
            self.chunks = chunks
            
            if embeddings is not None:
                self.embeddings = embeddings
            else:
                # Generate embeddings
                texts = [chunk.get('text', '') for chunk in chunks]
                self.embeddings = self.embedding_provider.encode(texts)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            if self.config.normalize_embeddings:
                faiss.normalize_L2(self.embeddings)
            
            self.index.add(self.embeddings.astype(np.float32))
            
            logger.info(f"Search index built with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load search data: {e}")
            raise
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index or not self.chunks:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_provider.encode([query])
            
            if self.config.normalize_embeddings:
                faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    result = self.chunks[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def is_ready(self) -> bool:
        """Check if search provider is ready"""
        return self.index is not None and len(self.chunks) > 0

class ImageProcessor:
    """Image processing and management"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.image_dirs = config.images_dirs
    
    def extract_images_from_results(self, search_results: List[Dict]) -> List[Dict]:
        """Extract and process images from search results"""
        images = []
        
        try:
            for result in search_results:
                result_images = result.get('images', [])
                for img_data in result_images:
                    # Process image data
                    processed_img = self._process_image_data(img_data, result)
                    if processed_img:
                        images.append(processed_img)
            
            # Limit number of images
            return images[:self.config.max_images_per_response]
            
        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return []
        
    def get_available_images(self) -> List[str]:
        available_images = []
    
        for img_dir in self.image_dirs:
            if os.path.exists(img_dir):
                try:
                    for file in os.listdir(img_dir):
                        if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                            available_images.append(os.path.join(img_dir, file))
                except Exception as e:
                    logger.warning(f"Erreur lecture répertoire {img_dir}: {e}")
    
        return available_images
    
    
    def _process_image_data(self, img_data: Dict, search_result: Dict) -> Optional[Dict]:
        """Process individual image data"""
        try:
            processed = {
                'path': img_data.get('path', ''),
                'description': img_data.get('description', ''),
                'enhanced_description': img_data.get('enhanced_description', img_data.get('description', '')),
                'stroke_focus': search_result.get('stroke', 'general'),
                'source': search_result.get('pdf', ''),
                'similarity_score': search_result.get('similarity_score', 0)
            }
            
            return processed
            
        except Exception as e:
            logger.warning(f"Image processing error: {e}")
            return None

class UIComponents:
    """UI component utilities"""

    @staticmethod
    def _get_badge_color(performance_class: str) -> str:
        """Get color for performance badge"""
        colors = {
            'badge-excellent': '#4caf50',
            'badge-good': '#2196f3',
            'badge-medium': '#ff9800',
            'badge-poor': '#f44336'
        }
        return colors.get(performance_class, '#9e9e9e')

    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color based on score"""
        if score >= 80:
            return '#4caf50'
        elif score >= 65:
            return '#2196f3'
        elif score >= 50:
            return '#ff9800'
        else:
            return '#f44336'
        


    @staticmethod
    def _render_images(images: List[Dict], image_processor , chat_id, question):
        "Render images avec indicateurs de génération LLM"""
        if not images:
            return
    
        st.markdown("### 🖼️ Visual Demonstrations")
    
        # Compter les descriptions générées par LLM
        llm_generated = sum(1 for img in images if img.get('generated_by_llm', False))
        if llm_generated > 0:
            st.info(f"🤖 {llm_generated} description(s) générée(s) par IA sur {len(images)}")
    
    # Fonction de recherche d'images (gardez la même)
        def find_image_in_cleaned_dirs(img_path: str) -> Optional[str]:
            if not img_path:
                return None
        
            filename = os.path.basename(img_path)
            cleaned_dirs = [
                "images_cleaned",
                "output/images_cleaned", 
                "output1/images_cleaned",
                "data/images_cleaned",
                "files/images_cleaned"
        ]   
        
            for directory in cleaned_dirs:
                if os.path.exists(directory):
                    full_path = os.path.join(directory, filename)
                    if os.path.exists(full_path):
                        return full_path
        
            if "images_cleaned/" in img_path:
                for directory in cleaned_dirs:
                    if os.path.exists(directory):
                        relative_path = img_path.split("images_cleaned/", 1)[1]
                        full_path = os.path.join(directory, relative_path)
                        if os.path.exists(full_path):
                            return full_path
        
            if os.path.exists(img_path):
                return img_path
            
            return None
    
        cols = st.columns(min(len(images), 3))
        images_displayed = 0
    
        for i, img in enumerate(images):
            with cols[i % 3]:
                img_path = img.get('path', '')
                description = img.get('description', img.get('enhanced_description', 'Swimming demonstration'))
                is_llm_generated = img.get('generated_by_llm', False)
            
                # Chercher le fichier image
                found_path = find_image_in_cleaned_dirs(img_path)
            
                if found_path:
                    try:
                        image = Image.open(found_path)
                        st.image(found_path, caption=description, use_container_width=True)
                        images_displayed += 1
                    
                        # Indicateur LLM
                        if is_llm_generated:
                            st.caption("🤖 Description générée par IA")
                    
                    except Exception as e:
                        st.info(f"📷 {description}")
                        st.text(f"Erreur: {str(e)[:50]}...")
                else:
                    st.info(f"📷 {description}")
                    st.text(f"Image: {os.path.basename(img_path) if img_path else 'Nom indisponible'}")
                    if is_llm_generated:
                        st.caption("🤖 Description IA")
            
                # Expander avec détails étendus
                with st.expander(f"Détails - Image {i+1}", expanded=False):
                    st.write(f"**Description:** {description}")
                    st.write(f"**Source:** {img.get('source', 'Inconnue')}")
                    st.write(f"**Type de nage:** {img.get('stroke_focus', 'Général')}")
                    st.write(f"**Nom fichier:** {img.get('filename', 'N/A')}")
                
                    if is_llm_generated:
                        st.success("🤖 Description générée par Intelligence Artificielle")
                    else:
                        st.info("📝 Description originale du document")
                
                    if img.get('similarity_score'):
                        st.write(f"**Pertinence:** {img.get('similarity_score', 0):.2f}")
                
                    # Bouton pour régénérer cette description
                    if st.button(f"🔄 Régénérer description",  key=f"regen_{chat_id}_{i}_{hash(question)}"
):
                        st.info("Régénération en cours...")
                        # Cette fonctionnalité sera implémentée dans l'interface principale
    
        # Statut final
        if images_displayed > 0:
            st.success(f"✅ {images_displayed} image(s) chargée(s) sur {len(images)}")
        else:
            st.warning(f"⚠️ Images non chargées mais {len(images)} description(s) disponible(s)")
    


    
    

    @staticmethod
    def _render_sources(sources: List[str]):
        """Render source information"""
        if not sources:
            return
        with st.expander("📚 Sources", expanded=False):
            for source in sources:
                st.markdown(f"- {source}")

    @staticmethod
    def _render_technical_details(search_results: List[Dict]):
        """Render technical details"""
        with st.expander("🔍 Technical Details", expanded=False):
            for i, result in enumerate(search_results[:3], 1):
                st.markdown(f"**Result {i}:**")
                st.markdown(f"- Source: {result.get('pdf', 'Unknown')}")
                st.markdown(f"- Similarity: {result.get('similarity_score', 0):.3f}")
                st.markdown(f"- Type: {result.get('swimming_type', 'General')}")
                if result.get('rerank_score'):
                    st.markdown(f"- Rerank Score: {result.get('rerank_score', 0):.3f}")
                st.markdown("---")

# ==================== REFERENCE DATASET MANAGEMENT ====================

class ReferenceDatasetManager:
    """Manages reference Q&A dataset for evaluation"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.reference_data = []
        self.load_reference_dataset()
    
    def load_reference_dataset(self):
        """Load or create reference dataset"""
        try:
            if os.path.exists(self.config.reference_dataset_path):
                with open(self.config.reference_dataset_path, 'r', encoding='utf-8') as f:
                    self.reference_data = json.load(f)
            else:
                self.reference_data = self._create_default_dataset()
                self.save_reference_dataset()
            
            logger.info(f"Loaded {len(self.reference_data)} reference Q&A pairs")
            
        except Exception as e:
            logger.error(f"Failed to load reference dataset: {e}")
            self.reference_data = self._create_default_dataset()
    
    def _create_default_dataset(self) -> List[Dict]:
        """Create default reference dataset"""
        return [
            {
                "id": 1,
                "question": "How to improve freestyle breathing technique?",
                "reference_answer": "To improve freestyle breathing, practice bilateral breathing (breathing on both sides), maintain body rotation, exhale underwater completely, and keep your head position low when breathing.",
                "key_concepts": ["bilateral breathing", "body rotation", "underwater exhale", "head position"],
                "stroke_type": "freestyle",
                "difficulty": "intermediate",
                "expected_sources": ["technique", "breathing"]
            },
            {
                "id": 2,
                "question": "What is the proper butterfly stroke technique?",
                "reference_answer": "Proper butterfly technique requires simultaneous arm movement, dolphin kick with core engagement, rhythmic breathing after every 2-3 strokes, and undulating body motion from chest to hips.",
                "key_concepts": ["simultaneous arms", "dolphin kick", "core engagement", "undulation", "rhythmic breathing"],
                "stroke_type": "butterfly",
                "difficulty": "advanced",
                "expected_sources": ["technique", "coordination"]
            },
            {
                "id": 3,
                "question": "How to increase swimming endurance?",
                "reference_answer": "Build swimming endurance through interval training, proper pacing, consistent technique, progressive distance increases, and incorporating different strokes to prevent overuse injuries.",
                "key_concepts": ["interval training", "pacing", "technique consistency", "progressive overload"],
                "stroke_type": "general",
                "difficulty": "intermediate",
                "expected_sources": ["training", "endurance"]
            },
            {
                "id": 4,
                "question": "What are effective backstroke drills?",
                "reference_answer": "Effective backstroke drills include single-arm backstroke, backstroke with flutter board, rotation drills, and catch-up backstroke to improve technique and body position.",
                "key_concepts": ["single-arm drill", "rotation", "body position", "catch-up"],
                "stroke_type": "backstroke",
                "difficulty": "intermediate", 
                "expected_sources": ["drills", "technique"]
            },
            {
                "id": 5,
                "question": "How to coordinate breaststroke timing?",
                "reference_answer": "Breaststroke coordination follows pull-breathe-kick-glide timing: pull arms, lift head to breathe, execute frog kick, then glide with arms extended.",
                "key_concepts": ["pull-breathe-kick-glide", "timing", "frog kick", "glide phase"],
                "stroke_type": "breaststroke",
                "difficulty": "advanced",
                "expected_sources": ["technique", "timing"]
            }
        ]
    
    def save_reference_dataset(self):
        """Save reference dataset"""
        try:
            with open(self.config.reference_dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.reference_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save reference dataset: {e}")
    
    def get_similar_reference(self, query: str) -> Optional[Dict]:
        """Find most similar reference question"""
        if not self.reference_data:
            return None
        
        query_words = set(query.lower().split())
        best_match = None
        best_score = 0
        
        for ref in self.reference_data:
            ref_words = set(ref['question'].lower().split())
            
            # Jaccard similarity
            intersection = len(query_words.intersection(ref_words))
            union = len(query_words.union(ref_words))
            
            if union > 0:
                score = intersection / union
                
                # Boost for concept overlap
                concept_overlap = sum(1 for concept in ref['key_concepts'] 
                                    if any(word in query.lower() for word in concept.lower().split()))
                score += concept_overlap * 0.1
                
                if score > best_score:
                    best_score = score
                    best_match = ref
        
        return best_match if best_score > 0.2 else None
    
    def add_reference(self, question: str, answer: str, concepts: List[str], 
                     stroke_type: str = "general", difficulty: str = "intermediate"):
        """Add new reference Q&A pair"""
        new_ref = {
            "id": len(self.reference_data) + 1,
            "question": question,
            "reference_answer": answer,
            "key_concepts": concepts,
            "stroke_type": stroke_type,
            "difficulty": difficulty,
            "created_at": datetime.now().isoformat()
        }
        
        self.reference_data.append(new_ref)
        self.save_reference_dataset()

# ==================== USER FEEDBACK SYSTEM ====================

class UserFeedbackManager:
    """Advanced user feedback collection and analysis"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.db_path = config.feedback_db_path
        self.init_database()
    
    def init_database(self):
        """Initialize feedback database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT,
                    response_text TEXT,
                    rating INTEGER,
                    helpfulness INTEGER,
                    accuracy INTEGER,
                    completeness INTEGER,
                    feedback_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    response_time_seconds REAL,
                    images_shown INTEGER,
                    sources_count INTEGER,
                    evaluation_score REAL
                )
            ''')
            
            # Interaction tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    action_type TEXT,
                    action_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feedback database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
    
    def record_feedback(self, session_id: str, question: str, response_data: Dict, 
                       rating: int, helpfulness: int, accuracy: int, 
                       completeness: int, feedback_text: str = ""):
        """Record user feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    session_id, question, response_text, rating, helpfulness, 
                    accuracy, completeness, feedback_text, response_time_seconds,
                    images_shown, sources_count, evaluation_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, question, response_data.get('text', ''),
                rating, helpfulness, accuracy, completeness, feedback_text,
                response_data.get('response_time', 0),
                len(response_data.get('images', [])),
                len(response_data.get('sources', [])),
                response_data.get('evaluation', {}).get('overall_score', 0)
            ))
            
            conn.commit()
            conn.close()
            logger.info("Feedback recorded successfully")
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    def record_interaction(self, session_id: str, action_type: str, action_data: Dict):
        """Record user interaction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO interactions (session_id, action_type, action_data)
                VALUES (?, ?, ?)
            ''', (session_id, action_type, json.dumps(action_data)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
    
    def get_feedback_analytics(self) -> Dict:
        """Get feedback analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Basic stats
            df_feedback = pd.read_sql_query('''
                SELECT rating, helpfulness, accuracy, completeness, 
                       images_shown, sources_count, evaluation_score,
                       DATE(timestamp) as feedback_date
                FROM feedback 
                WHERE timestamp >= date('now', '-30 days')
            ''', conn)
            
            analytics = {
                'total_feedback': len(df_feedback),
                'avg_rating': df_feedback['rating'].mean() if len(df_feedback) > 0 else 0,
                'avg_helpfulness': df_feedback['helpfulness'].mean() if len(df_feedback) > 0 else 0,
                'avg_accuracy': df_feedback['accuracy'].mean() if len(df_feedback) > 0 else 0,
                'avg_completeness': df_feedback['completeness'].mean() if len(df_feedback) > 0 else 0,
                'feedback_by_date': df_feedback.groupby('feedback_date').size().to_dict() if len(df_feedback) > 0 else {},
                'rating_distribution': df_feedback['rating'].value_counts().to_dict() if len(df_feedback) > 0 else {},
                'correlation_eval_rating': df_feedback[['evaluation_score', 'rating']].corr().iloc[0,1] if len(df_feedback) > 1 else 0
            }
            
            conn.close()
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {'total_feedback': 0, 'avg_rating': 0}
        

# ==================== HALLUCINATION DETECTION ===================

class HallucinationDetector:
    """Advanced hallucination detection system"""
    
    def __init__(self, config: RAGConfig, embedding_provider):
        self.config = config
        self.embedding_provider = embedding_provider
        
        # Swimming fact base
        self.swimming_facts = {
            'strokes': {
                'freestyle': ['front crawl', 'bilateral breathing', 'high elbow catch', 'body rotation'],
                'backstroke': ['back crawl', 'hip rotation', 'straight arm recovery', 'flutter kick'],
                'breaststroke': ['frog kick', 'pull-breathe-kick-glide', 'simultaneous movement', 'wide pull'],
                'butterfly': ['dolphin kick', 'undulation', 'simultaneous arms', 'rhythmic breathing']
            },
            'techniques': {
                'breathing': ['bilateral', 'rhythmic', 'underwater exhale', 'head position'],
                'kicking': ['flutter', 'dolphin', 'frog', 'ankle flexibility'],
                'pulling': ['high elbow', 'catch', 'pull pattern', 'hand entry']
            },
            'training': {
                'endurance': ['aerobic capacity', 'lactate threshold', 'distance per stroke'],
                'speed': ['anaerobic power', 'stroke rate', 'turn efficiency'],
                'technique': ['drills', 'stroke count', 'efficiency', 'body position']
            },
            'contraindications': [
                'breath holding underwater for extended periods',
                'hyperventilation before swimming',
                'swimming alone without supervision',
                'ignoring pool safety rules'
            ]
        }
        
        # Initialize fact checker if available
        self.fact_checker = None
        try:
            # This would be a more sophisticated fact-checking model in production
            self.fact_checker = pipeline("question-answering", 
                                       model="deepset/roberta-base-squad2",
                                       device=-1)  # CPU
        except Exception as e:
            logger.warning(f"Fact checker not available: {e}")
    
    def detect_hallucinations(self, response: str, context: List[Dict], 
                            query: str) -> Tuple[float, Dict]:
        """Comprehensive hallucination detection"""
        details = {
            'context_alignment': 0.0,
            'fact_consistency': 0.0,
            'swimming_domain_adherence': 0.0,
            'contradiction_detected': False,
            'unsupported_claims': [],
            'confidence_score': 0.0
        }
        
        try:
            # 1. Context alignment check
            context_score = self._check_context_alignment(response, context)
            details['context_alignment'] = context_score
            
            # 2. Fact consistency check
            fact_score = self._check_fact_consistency(response)
            details['fact_consistency'] = fact_score
            
            # 3. Domain adherence
            domain_score = self._check_domain_adherence(response)
            details['swimming_domain_adherence'] = domain_score
            
            # 4. Contradiction detection
            contradictions = self._detect_contradictions(response, context)
            details['contradiction_detected'] = len(contradictions) > 0
            details['contradictions'] = contradictions
            
            # 5. Unsupported claims
            unsupported = self._identify_unsupported_claims(response, context)
            details['unsupported_claims'] = unsupported
            
            # Overall hallucination risk score (lower is better)
            hallucination_risk = 1.0 - (
                context_score * 0.4 + 
                fact_score * 0.3 + 
                domain_score * 0.3
            )
            
            # Adjust for contradictions and unsupported claims
            if details['contradiction_detected']:
                hallucination_risk += 0.2
            
            hallucination_risk += len(unsupported) * 0.1
            hallucination_risk = min(hallucination_risk, 1.0)
            
            details['confidence_score'] = 1.0 - hallucination_risk
            
            return hallucination_risk, details
            
        except Exception as e:
            logger.error(f"Hallucination detection error: {e}")
            return 0.5, details
    
    def _check_context_alignment(self, response: str, context: List[Dict]) -> float:
        """Check alignment with provided context"""
        if not context:
            return 0.3  # Low score if no context provided
        
        response_words = set(response.lower().split())
        context_words = set()
        
        for chunk in context:
            context_words.update(chunk.get('text', '').lower().split())
        
        if not context_words or not response_words:
            return 0.3
        
        # Semantic overlap
        overlap = len(response_words.intersection(context_words))
        alignment_score = min(overlap / len(response_words), 1.0)
        
        # Boost for specific swimming terms
        swimming_terms_in_response = sum(1 for category in self.swimming_facts.values()
                                       for terms in category.values() if isinstance(terms, list)
                                       for term in terms if term in response.lower())
        
        alignment_score += min(swimming_terms_in_response * 0.05, 0.3)
        
        return min(alignment_score, 1.0)
    
    def _check_fact_consistency(self, response: str) -> float:
        """Check factual consistency against swimming knowledge"""
        response_lower = response.lower()
        
        # Check for swimming facts
        fact_score = 0.5  # Base score
        
        # Positive indicators
        for category, subcategories in self.swimming_facts.items():
            if category == 'contraindications':
                continue
                
            for subcategory, facts in subcategories.items():
                if isinstance(facts, list):
                    for fact in facts:
                        if fact in response_lower:
                            fact_score += 0.05
        
        # Negative indicators (contraindications or dangerous advice)
        for contraindication in self.swimming_facts['contraindications']:
            if contraindication in response_lower:
                fact_score -= 0.3
        
        # Check for impossible claims
        impossible_patterns = [
            r'swim.*\d+.*miles.*without.*training',
            r'hold.*breath.*\d+.*minutes',
            r'learn.*swimming.*\d+.*days?',
            r'guarantee.*\d+.*seconds.*improvement'
        ]
        
        for pattern in impossible_patterns:
            if re.search(pattern, response_lower):
                fact_score -= 0.2
        
        return max(min(fact_score, 1.0), 0.0)
    
    def _check_domain_adherence(self, response: str) -> float:
        """Check adherence to swimming domain"""
        response_lower = response.lower()
        
        # Swimming-related terms
        swimming_terms = ['swim', 'stroke', 'pool', 'water', 'technique', 'training',
                         'freestyle', 'backstroke', 'breaststroke', 'butterfly',
                         'kick', 'pull', 'breathing', 'drill', 'lap', 'lane']
        
        swimming_term_count = sum(1 for term in swimming_terms if term in response_lower)
        
        # Calculate domain relevance
        word_count = len(response_lower.split())
        if word_count == 0:
            return 0.0
        
        domain_density = swimming_term_count / word_count
        domain_score = min(domain_density * 10, 1.0)  # Scale up
        
        # Boost for technical accuracy indicators
        technical_terms = ['technique', 'form', 'efficiency', 'coordination', 'timing']
        technical_bonus = sum(0.1 for term in technical_terms if term in response_lower)
        
        return min(domain_score + technical_bonus, 1.0)
    
    def _detect_contradictions(self, response: str, context: List[Dict]) -> List[str]:
        """Detect contradictions between response and context"""
        contradictions = []
        
        # Simple contradiction patterns
        response_lower = response.lower()
        
        # Extract context statements
        context_statements = []
        for chunk in context:
            text = chunk.get('text', '')
            # Simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            context_statements.extend(sentences[:5])  # Limit for performance
        
        # Check for direct contradictions
        contradiction_patterns = [
            ('always', 'never'),
            ('increase', 'decrease'),
            ('faster', 'slower'),
            ('more', 'less'),
            ('should', 'should not'),
            ('recommended', 'not recommended')
        ]
        
        for context_stmt in context_statements:
            context_lower = context_stmt.lower()
            for pattern1, pattern2 in contradiction_patterns:
                if pattern1 in context_lower and pattern2 in response_lower:
                    contradictions.append(f"Context says '{pattern1}' but response suggests '{pattern2}'")
                elif pattern2 in context_lower and pattern1 in response_lower:
                    contradictions.append(f"Context says '{pattern2}' but response suggests '{pattern1}'")
        
        return contradictions[:3]  # Limit output
    
    def _identify_unsupported_claims(self, response: str, context: List[Dict]) -> List[str]:
        """Identify claims not supported by context"""
        unsupported = []
        
        # Extract factual claims from response
        claim_patterns = [
            r'studies show that ([^.]+)',
            r'research indicates ([^.]+)', 
            r'experts recommend ([^.]+)',
            r'the best way to ([^.]+) is ([^.]+)',
            r'([^.]+) improves ([^.]+) by (\d+)%'
        ]
        
        context_text = ' '.join(chunk.get('text', '') for chunk in context).lower()
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, response.lower())
            for match in matches:
                claim = match.group(0)
                # Check if claim has any support in context
                claim_words = set(claim.split())
                context_words = set(context_text.split())
                
                support_ratio = len(claim_words.intersection(context_words)) / len(claim_words)
                if support_ratio < 0.3:  # Low support threshold
                    unsupported.append(claim[:100] + "..." if len(claim) > 100 else claim)
        
        return unsupported[:3]  # Limit output

# ==================== ADVANCED RERANKING SYSTEM ====================

class AdvancedReranker:
    """Sophisticated reranking with multiple signals"""
    
    def __init__(self, config: RAGConfig, embedding_provider):
        self.config = config
        self.embedding_provider = embedding_provider
        
        # Initialize TF-IDF for lexical matching
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_fitted = False
    
    def rerank_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Advanced reranking with multiple signals"""
        if not search_results or len(search_results) <= 1:
            return search_results
        
        try:
            # Prepare texts for reranking
            texts = [result.get('text', '') for result in search_results]
            
            # Calculate multiple ranking signals
            semantic_scores = self._calculate_semantic_scores(query, texts)
            lexical_scores = self._calculate_lexical_scores(query, texts)
            domain_scores = self._calculate_domain_scores(texts)
            freshness_scores = self._calculate_freshness_scores(search_results)
            completeness_scores = self._calculate_completeness_scores(texts)
            
            # Combine signals with weights
            final_scores = []
            for i in range(len(search_results)):
                combined_score = (
                    semantic_scores[i] * 0.35 +
                    lexical_scores[i] * 0.25 +
                    domain_scores[i] * 0.20 +
                    completeness_scores[i] * 0.15 +
                    freshness_scores[i] * 0.05
                )
                final_scores.append(combined_score)
            
            # Rerank based on combined scores
            ranked_indices = sorted(range(len(search_results)), 
                                  key=lambda i: final_scores[i], reverse=True)
            
            reranked_results = []
            for idx in ranked_indices:
                result = search_results[idx].copy()
                result['rerank_score'] = final_scores[idx]
                result['rerank_details'] = {
                    'semantic': semantic_scores[idx],
                    'lexical': lexical_scores[idx], 
                    'domain': domain_scores[idx],
                    'completeness': completeness_scores[idx],
                    'freshness': freshness_scores[idx]
                }
                reranked_results.append(result)
            
            return reranked_results[:self.config.top_k_retrieval]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return search_results
    
    def _calculate_semantic_scores(self, query: str, texts: List[str]) -> List[float]:
        """Calculate semantic similarity scores"""
        try:
            if self.embedding_provider.is_available():
                query_embedding = self.embedding_provider.encode([query])
                text_embeddings = self.embedding_provider.encode(texts)
                similarities = cosine_similarity(query_embedding, text_embeddings)[0]
                return similarities.tolist()
            else:
                return [0.5] * len(texts)
        except Exception as e:
            logger.warning(f"Semantic scoring error: {e}")
            return [0.5] * len(texts)
    
    def _calculate_lexical_scores(self, query: str, texts: List[str]) -> List[float]:
        """Calculate lexical overlap scores"""
        try:
            all_texts = [query] + texts
            
            if not self.tfidf_fitted:
                self.tfidf.fit(all_texts)
                self.tfidf_fitted = True
            
            tfidf_matrix = self.tfidf.transform(all_texts)
            query_vector = tfidf_matrix[0]
            text_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, text_vectors)[0]
            return similarities.tolist()
            
        except Exception as e:
            logger.warning(f"Lexical scoring error: {e}")
            # Fallback to simple word overlap
            query_words = set(query.lower().split())
            scores = []
            for text in texts:
                text_words = set(text.lower().split())
                if query_words and text_words:
                    overlap = len(query_words.intersection(text_words)) / len(query_words)
                else:
                    overlap = 0
                scores.append(overlap)
            return scores
    
    def _calculate_domain_scores(self, texts: List[str]) -> List[float]:
        """Calculate swimming domain relevance scores"""
        swimming_keywords = {
            'strokes': ['freestyle', 'backstroke', 'breaststroke', 'butterfly', 'stroke'],
            'technique': ['technique', 'form', 'mechanics', 'coordination', 'timing'],
            'training': ['training', 'workout', 'drill', 'practice', 'exercise'],
            'performance': ['speed', 'endurance', 'efficiency', 'improvement']
        }
        
        scores = []
        for text in texts:
            text_lower = text.lower()
            domain_score = 0
            
            for category, keywords in swimming_keywords.items():
                category_count = sum(1 for keyword in keywords if keyword in text_lower)
                domain_score += category_count * 0.1
            
            # Normalize by text length
            word_count = len(text_lower.split())
            if word_count > 0:
                domain_score = min(domain_score / word_count * 100, 1.0)
            
            scores.append(domain_score)
        
        return scores
    
    def _calculate_freshness_scores(self, search_results: List[Dict]) -> List[float]:
        """Calculate content freshness scores"""
        # In a real system, this would use document timestamps
        # For now, we'll use document diversity as a proxy
        sources = [result.get('pdf', 'unknown') for result in search_results]
        unique_sources = set(sources)
        
        scores = []
        for result in search_results:
            source = result.get('pdf', 'unknown')
            # Penalize over-representation of same source
            source_count = sources.count(source)
            freshness = max(0.3, 1.0 - (source_count - 1) * 0.2)
            scores.append(freshness)
        
        return scores
    
    def _calculate_completeness_scores(self, texts: List[str]) -> List[float]:
        """Calculate content completeness scores"""
        scores = []
        for text in texts:
            # Length-based completeness (longer = more complete, up to a point)
            length_score = min(len(text) / 1000, 1.0)
            
            # Structure indicators
            structure_indicators = ['.', ':', '-', '\n', '1.', '2.', 'first', 'second']
            structure_count = sum(1 for indicator in structure_indicators if indicator in text.lower())
            structure_score = min(structure_count * 0.1, 0.5)
            
            completeness = length_score * 0.7 + structure_score * 0.3
            scores.append(completeness)
        
        return scores









class LLMDescriptionGenerator:
    """Générateur de descriptions d'images utilisant un LLM"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.cache = {}  # Cache pour éviter les appels répétés
        self.max_daily_generations = 100  # Limite quotidienne
        self.generation_count = 0
    
    def generate_description_from_context(self, filename: str, stroke_focus: str, 
                                        context_text: str, source_pdf: str) -> str:
        """Génère une description d'image basée sur le contexte et le nom de fichier"""
        
        # Vérifier le cache
        cache_key = f"{filename}_{stroke_focus}_{hash(context_text[:200])}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            prompt = self._build_description_prompt(filename, stroke_focus, context_text, source_pdf)
            description = self.llm_provider.generate_response(prompt)
            
            # Nettoyer et valider la description
            clean_description = self._clean_description(description)
            
            # Mettre en cache
            self.cache[cache_key] = clean_description
            
            return clean_description
            
        except Exception as e:
            logger.error(f"Erreur génération description LLM: {e}")
            return self._fallback_description(filename, stroke_focus)
    
    def _build_description_prompt(self, filename: str, stroke_focus: str, 
                                context_text: str, source_pdf: str) -> str:
        """Construit le prompt pour la génération de description"""
        
        prompt = f"""Tu es un expert en natation. Génère une description précise et informative pour une image de démonstration de natation basée sur les informations suivantes :

FICHIER IMAGE: {filename}
TYPE DE NAGE: {stroke_focus}
SOURCE: {source_pdf}

CONTEXTE DOCUMENTAIRE:
{context_text[:500]}

INSTRUCTIONS:
- Génère une description de 1-2 phrases maximum
- Décris ce que l'image montre probablement (technique, exercice, position)
- Utilise un langage technique approprié pour la natation
- Sois spécifique au type de nage mentionné
- Reste factuel et informatif

EXEMPLES DE FORMAT:
- "Démonstration de la position de corps en nage libre avec rotation correcte"
- "Exercice de battement de jambes en brasse avec position des hanches"
- "Technique de respiration bilatérale en crawl"

Génère UNIQUEMENT la description, sans texte supplémentaire:"""

        return prompt
    
    def _clean_description(self, raw_description: str) -> str:
        """Nettoie et valide la description générée"""
        if not raw_description:
            return "Démonstration de natation"
        
        # Nettoyer le texte
        clean_desc = raw_description.strip()
        
        # Enlever les guillemets si présents
        if clean_desc.startswith('"') and clean_desc.endswith('"'):
            clean_desc = clean_desc[1:-1]
        
        # Limiter la longueur
        if len(clean_desc) > 150:
            clean_desc = clean_desc[:147] + "..."
        
        # Valider que c'est bien une description de natation
        swimming_keywords = ['natation', 'nage', 'crawl', 'brasse', 'dos', 'papillon', 
                           'technique', 'exercice', 'démonstration', 'position', 
                           'battement', 'respiration', 'stroke', 'swim']
        
        if not any(keyword in clean_desc.lower() for keyword in swimming_keywords):
            clean_desc = f"Technique de natation - {clean_desc}"
        
        return clean_desc
    
    def _fallback_description(self, filename: str, stroke_focus: str) -> str:
        """Description de fallback si le LLM échoue"""
        base_descriptions = {
            'freestyle': 'Technique de nage libre',
            'backstroke': 'Technique de dos crawlé',
            'breaststroke': 'Technique de brasse',
            'butterfly': 'Technique de papillon',
            'general': 'Technique de natation'
        }
        
        base = base_descriptions.get(stroke_focus, 'Technique de natation')
        
        # Ajouter des détails basés sur le nom de fichier
        if 'drill' in filename.lower():
            return f"Exercice d'entraînement - {base.lower()}"
        elif 'technique' in filename.lower():
            return f"Démonstration de {base.lower()}"
        else:
            return base

class EnhancedImageProcessor(ImageProcessor):
    """Version améliorée du processeur d'images avec génération LLM"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.description_generator = LLMDescriptionGenerator(config)
    
    def _process_image_data(self, img_data: Dict, search_result: Dict) -> Optional[Dict]:
        """Process individual image data avec génération LLM"""
        try:
            img_path = img_data.get('path', '')
            filename = os.path.basename(img_path) if img_path else ''
            stroke_focus = search_result.get('stroke', 'general')
            
            # Récupérer la description originale
            original_desc = img_data.get('enhanced_description') or img_data.get('description', '')
            
            # Si pas de description ou description basique, générer avec LLM
            if (not original_desc or 
                original_desc == "No description available" or 
                len(original_desc) < 10):
                
                # Utiliser le contexte du résultat de recherche pour la génération
                context_text = search_result.get('text', '')
                source_pdf = search_result.get('pdf', '')
                
                description = self.description_generator.generate_description_from_context(
                    filename, stroke_focus, context_text, source_pdf
                )
            else:
                description = original_desc
            
            processed = {
                'path': img_path,
                'description': description,
                'enhanced_description': description,
                'stroke_focus': stroke_focus,
                'source': search_result.get('pdf', ''),
                'similarity_score': search_result.get('similarity_score', 0),
                'filename': filename,
                'generated_by_llm': not original_desc or len(original_desc) < 10
            }
            
            return processed
            
        except Exception as e:
            logger.warning(f"Enhanced image processing error: {e}")
            # Fallback avec génération simple
            fallback_desc = self.description_generator._fallback_description(
                img_data.get('path', ''), search_result.get('stroke', 'general')
            )
            
            return {
                'path': img_data.get('path', ''),
                'description': fallback_desc,
                'enhanced_description': fallback_desc,
                'stroke_focus': search_result.get('stroke', 'general'),
                'source': search_result.get('pdf', ''),
                'similarity_score': 0,
                'filename': os.path.basename(img_data.get('path', '')),
                'generated_by_llm': True
            }

# Modification de la classe EnhancedSwimmingRAGSystem pour utiliser le nouveau processeur :

# Dans la méthode __init__, remplacez :
# self.image_processor = ImageProcessor(config)
# par :
#self.image_processor = EnhancedImageProcessor(config)

# Ajoutez aussi cette méthode pour régénérer les descriptions :

def regenerate_image_descriptions(self, search_results: List[Dict]) -> List[Dict]:
    """Régénère les descriptions d'images avec le LLM"""
    enhanced_results = []
    
    for result in search_results:
        enhanced_result = result.copy()
        images = result.get('images', [])
        
        if images:
            enhanced_images = []
            for img_data in images:
                # Force la régénération avec LLM
                enhanced_img = self.image_processor._process_image_data(img_data, result)
                if enhanced_img:
                    enhanced_images.append(enhanced_img)
            
            enhanced_result['images'] = enhanced_images
        
        enhanced_results.append(enhanced_result)
    
    return enhanced_results

# Interface utilisateur pour contrôler la génération LLM :

def render_llm_description_controls():
    """Interface de contrôle pour la génération de descriptions LLM"""
    
    with st.sidebar:
        st.markdown("### 🤖 Génération LLM")
        
        enable_llm_descriptions = st.checkbox(
            "Descriptions LLM", 
            value=st.session_state.get('enable_llm_descriptions', True),
            help="Utiliser le LLM pour générer des descriptions d'images intelligentes"
        )
        st.session_state.enable_llm_descriptions = enable_llm_descriptions
        
        if enable_llm_descriptions:
            force_regenerate = st.checkbox(
                "Forcer régénération",
                value=False,
                help="Régénérer toutes les descriptions même si elles existent déjà"
            )
            st.session_state.force_llm_regenerate = force_regenerate
            
            # Afficher le statut du cache
            if hasattr(st.session_state, 'enhanced_rag_system'):
                rag_system = st.session_state.enhanced_rag_system
                if hasattr(rag_system.image_processor, 'description_generator'):
                    cache_size = len(rag_system.image_processor.description_generator.cache)
                    st.metric("Cache descriptions", cache_size)
                    
                    if st.button("Vider cache"):
                        rag_system.image_processor.description_generator.cache.clear()
                        st.success("Cache vidé")
                        st.rerun()


class QueryExpander:
    """Advanced query expansion with context awareness"""
    
    def __init__(self, config: RAGConfig, embedding_provider):
        self.config = config
        self.embedding_provider = embedding_provider
        
        # Swimming domain synonyms
        self.synonyms = {
            'freestyle': ['front crawl', 'crawl stroke', 'free'],
            'backstroke': ['back crawl', 'back stroke', 'back'],
            'breaststroke': ['breast stroke', 'frog stroke'],
            'butterfly': ['fly stroke', 'dolphin stroke', 'fly'],
            'improve': ['enhance', 'better', 'develop', 'increase'],
            'technique': ['form', 'mechanics', 'method', 'style'],
            'training': ['workout', 'practice', 'drill', 'exercise'],
            'speed': ['fast', 'quick', 'velocity', 'pace'],
            'endurance': ['stamina', 'conditioning', 'aerobic'],
            'breathing': ['breath', 'ventilation', 'air', 'oxygen']
        }
        
        # Contextual expansions
        self.context_expansions = {
            'beginner': ['learn', 'basic', 'start', 'introduction'],
            'advanced': ['competitive', 'expert', 'master', 'elite'],
            'problem': ['fix', 'correct', 'solve', 'troubleshoot'],
            'timing': ['coordination', 'rhythm', 'sequence', 'synchronization']
        }
    
    def expand_query(self, query: str, user_context: Optional[Dict] = None) -> str:
        """Intelligently expand query with relevant terms"""
        if not self.config.enable_query_expansion:
            return query
        
        try:
            expanded_terms = []
            query_words = query.lower().split()
            
            # Add synonyms for swimming terms
            for word in query_words:
                if word in self.synonyms:
                    # Add most relevant synonym
                    best_synonym = self._select_best_synonym(word, query, user_context)
                    if best_synonym and best_synonym not in query.lower():
                        expanded_terms.append(best_synonym)
            
            # Add contextual terms
            context_terms = self._get_contextual_terms(query, user_context)
            expanded_terms.extend(context_terms)
            
            # Add technique-specific terms
            technique_terms = self._get_technique_terms(query)
            expanded_terms.extend(technique_terms)
            
            # Combine with original query
            if expanded_terms:
                expansion = ' '.join(expanded_terms[:3])  # Limit expansion
                expanded_query = f"{query} {expansion}"
                logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
                return expanded_query
            
            return query
            
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return query
    
    def _select_best_synonym(self, word: str, query: str, context: Optional[Dict]) -> str:
        """Select most appropriate synonym based on context"""
        if word not in self.synonyms:
            return ""
        
        synonyms = self.synonyms[word]
        if not synonyms:
            return ""
        
        # Simple heuristic: prefer longer terms for technical queries
        if any(tech_word in query.lower() for tech_word in ['technique', 'form', 'mechanics']):
            return max(synonyms, key=len)
        else:
            return synonyms[0]  # Default to first synonym
    
    def _get_contextual_terms(self, query: str, context: Optional[Dict]) -> List[str]:
        """Get contextually relevant expansion terms"""
        terms = []
        query_lower = query.lower()
        
        # Skill level context
        if any(word in query_lower for word in ['learn', 'begin', 'start', 'new']):
            terms.extend(self.context_expansions.get('beginner', [])[:2])
        elif any(word in query_lower for word in ['compete', 'race', 'fast', 'elite']):
            terms.extend(self.context_expansions.get('advanced', [])[:2])
        
        # Problem-solving context
        if any(word in query_lower for word in ['problem', 'wrong', 'error', 'fix', 'correct']):
            terms.extend(self.context_expansions.get('problem', [])[:2])
        
        return terms[:2]  # Limit contextual terms
    
    def _get_technique_terms(self, query: str) -> List[str]:
        """Get technique-specific expansion terms"""
        technique_map = {
            'breathing': ['bilateral', 'underwater', 'rhythm'],
            'kick': ['flutter', 'dolphin', 'frog', 'ankle'],
            'pull': ['catch', 'high elbow', 'stroke pattern'],
            'turn': ['flip', 'open', 'push off', 'streamline'],
            'start': ['dive', 'reaction time', 'entry']
        }
        
        query_lower = query.lower()
        terms = []
        
        for technique, related_terms in technique_map.items():
            if technique in query_lower:
                terms.extend(related_terms[:2])  # Add up to 2 related terms
        
        return terms[:2]  # Limit technique terms

# ==================== DEGRADED MODE SYSTEM ====================
class SystemStatus(Enum):
    """System status enumeration"""
    FULL = "full"
    ADVANCED = "advanced"
    BASIC = "basic"
    MINIMAL = "minimal"
    OFFLINE = "offline"
    ERROR = "error"

class DegradedModeManager:
    """Manages system degradation with coherent fallback modes"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.current_mode = SystemMode.FULL
        self.fallback_responses = self._initialize_fallback_responses()
        self.cached_responses = {}
    
    def _initialize_fallback_responses(self) -> Dict[str, str]:
        """Initialize fallback response templates"""
        return {
            'freestyle_technique': """
            For freestyle technique improvement:
            
            1. **Body Position**: Keep your body horizontal with head in neutral position
            2. **Arm Stroke**: High elbow catch, pull through to hip, rotate arms smoothly
            3. **Breathing**: Practice bilateral breathing (every 3 strokes)
            4. **Kick**: Steady flutter kick from hips, ankles flexible
            5. **Timing**: Coordinate arm strokes with body rotation
            
            Practice with focused drills and gradual progression.
            """,
            
            'backstroke_technique': """
            For backstroke improvement:
            
            1. **Body Position**: Float on back, hips high, head stable
            2. **Arm Movement**: Straight arm recovery, rotate shoulders
            3. **Kick**: Flutter kick, toes just breaking surface
            4. **Timing**: Consistent rhythm, opposite arm-leg coordination
            5. **Breathing**: Breathe freely, maintain steady rhythm
            
            Focus on rotation and maintaining straight body line.
            """,
            
            'breaststroke_timing': """
            For breaststroke coordination:
            
            1. **Sequence**: Pull - Breathe - Kick - Glide
            2. **Arm Pull**: Wide sweep, hands together at chest
            3. **Breathing**: Lift head naturally with arm pull
            4. **Kick**: Frog kick, heels to glutes, snap together
            5. **Glide**: Brief streamline position between strokes
            
            Practice each component separately before combining.
            """,
            
            'butterfly_coordination': """
            For butterfly stroke development:
            
            1. **Undulation**: Wave-like motion from chest through hips
            2. **Arms**: Simultaneous movement, enter together
            3. **Kick**: Two dolphin kicks per arm cycle
            4. **Breathing**: Every 2-3 strokes, quick head lift
            5. **Timing**: Second kick as hands enter water
            
            Start with dolphin kick drills and body wave practice.
            """,
            
            'general_training': """
            For swimming training improvement:
            
            1. **Consistency**: Regular practice schedule
            2. **Technique Focus**: Quality over quantity
            3. **Progressive Overload**: Gradually increase distance/intensity
            4. **Variety**: Mix different strokes and training types
            5. **Recovery**: Include rest days and easy sessions
            
            Build endurance gradually while maintaining proper form.
            """,
            
            'default': """
            Here are general swimming improvement principles:
            
            1. **Master the Basics**: Focus on body position and breathing
            2. **Practice Regularly**: Consistency builds muscle memory
            3. **Get Feedback**: Work with a coach or experienced swimmer
            4. **Use Drills**: Break down complex movements into components
            5. **Be Patient**: Swimming technique develops over time
            
            Consider taking lessons or joining a swim group for structured improvement.
            """
        }
    
    def determine_appropriate_mode(self, error_type: str, system_components: Dict) -> SystemMode:
        """Determine appropriate degraded mode based on system state"""
        
        # Check component availability
        embedding_available = system_components.get('embedding_provider', False)
        search_available = system_components.get('search_provider', False)
        llm_available = system_components.get('llm_provider', False)
        
        if all([embedding_available, search_available, llm_available]):
            return SystemMode.FULL
        elif search_available and llm_available:
            return SystemMode.ADVANCED
        elif llm_available:
            return SystemMode.BASIC
        elif embedding_available or search_available:
            return SystemMode.MINIMAL
        else:
            return SystemMode.OFFLINE
    
    def get_degraded_response(self, query: str, mode: SystemMode) -> Dict[str, Any]:
        """Generate response in degraded mode"""
        
        if mode == SystemMode.OFFLINE:
            return self._get_offline_response(query)
        elif mode == SystemMode.MINIMAL:
            return self._get_minimal_response(query)
        elif mode == SystemMode.BASIC:
            return self._get_basic_response(query)
        else:
            return self._get_fallback_response()
    
    def _get_offline_response(self, query: str) -> Dict[str, Any]:
        """Generate response when fully offline"""
        
        # Analyze query for best template match
        template_key = self._match_query_to_template(query)
        response_text = self.fallback_responses[template_key]
        
        return {
            'text': f"🔧 **System Operating in Offline Mode**\n\n{response_text}\n\n*This is a cached response. For personalized advice, please try again when the system is fully operational.*",
            'images': [],
            'sources': ['Offline Cache - General Swimming Knowledge'],
            'search_results': [],
            'evaluation': None,
            'system_status': SystemMode.OFFLINE.value,
            'degraded_mode': True
        }
    
    def _get_minimal_response(self, query: str) -> Dict[str, Any]:
        """Generate response in minimal mode"""
        
        template_key = self._match_query_to_template(query)
        base_response = self.fallback_responses[template_key]
        
        # Add basic personalization
        if 'beginner' in query.lower() or 'learn' in query.lower():
            personalization = "\n\n**For Beginners**: Start with basic drills and focus on comfort in water before advancing to technique refinement."
        elif 'competitive' in query.lower() or 'race' in query.lower():
            personalization = "\n\n**For Competitive Swimming**: Focus on efficiency, stroke rate optimization, and race-specific training."
        else:
            personalization = "\n\n**General Advice**: Adapt these suggestions to your current skill level and practice regularly."
        
        return {
            'text': f"⚠️ **System Operating in Minimal Mode**\n\n{base_response}{personalization}",
            'images': [],
            'sources': ['Minimal Mode - Basic Swimming Guidance'],
            'search_results': [],
            'evaluation': {'overall_score': 40.0, 'performance_level': 'Degraded Mode', 'performance_class': 'badge-medium'},
            'system_status': SystemMode.MINIMAL.value,
            'degraded_mode': True
        }
    
    def _get_basic_response(self, query: str) -> Dict[str, Any]:
        """Generate response in basic mode with simple processing"""
        
        # Simple keyword-based response enhancement
        query_words = set(query.lower().split())
        
        template_key = self._match_query_to_template(query)
        base_response = self.fallback_responses[template_key]
        
        # Add context-aware additions
        enhancements = []
        
        if 'improve' in query_words or 'better' in query_words:
            enhancements.append("**Key Focus**: Consistent practice and gradual progression are essential for improvement.")
        
        if 'breathing' in query_words:
            enhancements.append("**Breathing Tip**: Practice rhythmic breathing and complete underwater exhalation.")
        
        if 'speed' in query_words or 'fast' in query_words:
            enhancements.append("**Speed Development**: Focus on technique first, then gradually increase stroke rate.")
        
        enhancement_text = "\n\n" + "\n".join(enhancements) if enhancements else ""
        
        return {
            'text': f"🔧 **System Operating in Basic Mode**\n\n{base_response}{enhancement_text}\n\n*Limited functionality active. Full system features unavailable.*",
            'images': [],
            'sources': ['Basic Mode - Enhanced Templates'],
            'search_results': [],
            'evaluation': {'overall_score': 55.0, 'performance_level': 'Basic Mode', 'performance_class': 'badge-medium'},
            'system_status': SystemMode.BASIC.value,
            'degraded_mode': True
        }
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get basic fallback response"""
        return {
            'text': "The system is experiencing technical difficulties. Please try rephrasing your question or contact support if the issue persists.",
            'images': [],
            'sources': [],
            'search_results': [],
            'evaluation': None,
            'system_status': SystemMode.ERROR.value,
            'degraded_mode': True
        }
    
    def _match_query_to_template(self, query: str) -> str:
        """Match query to most appropriate template"""
        query_lower = query.lower()
        
        # Stroke-specific matching
        if 'freestyle' in query_lower or 'front crawl' in query_lower:
            return 'freestyle_technique'
        elif 'backstroke' in query_lower or 'back' in query_lower:
            return 'backstroke_technique'
        elif 'breaststroke' in query_lower or 'breast' in query_lower:
            return 'breaststroke_timing'
        elif 'butterfly' in query_lower or 'fly' in query_lower:
            return 'butterfly_coordination'
        elif any(word in query_lower for word in ['training', 'workout', 'endurance', 'fitness']):
            return 'general_training'
        else:
            return 'default'

# ==================== REFERENCE-BASED EVALUATION ====================

class ReferenceBasedEvaluator:
    """Evaluation against reference dataset"""
    
    def __init__(self, reference_manager: ReferenceDatasetManager, embedding_provider):
        self.reference_manager = reference_manager
        self.embedding_provider = embedding_provider
    
    def evaluate_against_reference(self, query: str, response: str, 
                                 search_results: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate response against reference dataset"""
        
        # Find similar reference
        reference = self.reference_manager.get_similar_reference(query)
        if not reference:
            return 50.0, {'reference_found': False, 'message': 'No suitable reference found'}
        
        details = {
            'reference_found': True,
            'reference_id': reference['id'],
            'reference_similarity': 0.0,
            'concept_coverage': 0.0,
            'answer_similarity': 0.0,
            'source_alignment': 0.0,
            'overall_reference_score': 0.0
        }
        
        try:
            # 1. Query-Reference similarity
            query_similarity = self._calculate_query_similarity(query, reference['question'])
            details['reference_similarity'] = query_similarity
            
            # 2. Concept coverage
            concept_coverage = self._evaluate_concept_coverage(response, reference['key_concepts'])
            details['concept_coverage'] = concept_coverage
            
            # 3. Answer similarity
            answer_similarity = self._evaluate_answer_similarity(response, reference['reference_answer'])
            details['answer_similarity'] = answer_similarity
            
            # 4. Expected source alignment
            source_alignment = self._evaluate_source_alignment(search_results, reference.get('expected_sources', []))
            details['source_alignment'] = source_alignment
            
            # Overall score with weights
            overall_score = (
                concept_coverage * 0.4 +
                answer_similarity * 0.3 +
                source_alignment * 0.2 +
                query_similarity * 0.1
            )
            
            details['overall_reference_score'] = overall_score
            
            return overall_score, details
            
        except Exception as e:
            logger.error(f"Reference evaluation error: {e}")
            return 50.0, details
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between queries"""
        try:
            if self.embedding_provider.is_available():
                embeddings = self.embedding_provider.encode([query1, query2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity) * 100
            else:
                # Fallback to word overlap
                words1 = set(query1.lower().split())
                words2 = set(query2.lower().split())
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    return overlap * 100
                return 0.0
        except Exception as e:
            logger.warning(f"Query similarity error: {e}")
            return 50.0
    
    def _evaluate_concept_coverage(self, response: str, key_concepts: List[str]) -> float:
        """Evaluate how well response covers key concepts"""
        response_lower = response.lower()
        covered_concepts = 0
        
        for concept in key_concepts:
            concept_words = concept.lower().split()
            # Check if all words of concept appear in response
            if all(word in response_lower for word in concept_words):
                covered_concepts += 1
            # Partial credit for partial matches
            elif any(word in response_lower for word in concept_words):
                covered_concepts += 0.5
        
        if key_concepts:
            coverage_score = (covered_concepts / len(key_concepts)) * 100
        else:
            coverage_score = 50.0
        
        return min(coverage_score, 100.0)
    
    def _evaluate_answer_similarity(self, response: str, reference_answer: str) -> float:
        """Evaluate similarity to reference answer"""
        try:
            if self.embedding_provider.is_available():
                embeddings = self.embedding_provider.encode([response, reference_answer])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity) * 100
            else:
                # Fallback to lexical similarity
                response_words = set(response.lower().split())
                reference_words = set(reference_answer.lower().split())
                
                if response_words and reference_words:
                    intersection = len(response_words.intersection(reference_words))
                    union = len(response_words.union(reference_words))
                    similarity = intersection / union if union > 0 else 0
                    return similarity * 100
                return 0.0
        except Exception as e:
            logger.warning(f"Answer similarity error: {e}")
            return 50.0
    
    def _evaluate_source_alignment(self, search_results: List[Dict], 
                                 expected_sources: List[str]) -> float:
        """Evaluate alignment with expected source types"""
        if not expected_sources:
            return 70.0  # Default score if no expectations
        
        found_sources = set()
        for result in search_results:
            # Extract source types from result
            text_lower = result.get('text', '').lower()
            pdf_name = result.get('pdf', '').lower()
            swim_type = result.get('swimming_type', '').lower()
            
            for expected in expected_sources:
                if (expected.lower() in text_lower or 
                    expected.lower() in pdf_name or 
                    expected.lower() in swim_type):
                    found_sources.add(expected.lower())
        
        if expected_sources:
            alignment_score = (len(found_sources) / len(expected_sources)) * 100
        else:
            alignment_score = 70.0
        
        return min(alignment_score, 100.0)

# ==================== ENHANCED MAIN RAG SYSTEM ====================

class EnhancedSwimmingRAGSystem:
    """Enhanced RAG system with all improvements"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.system_mode = SystemMode.FULL

        self.system_ready = False
        self.component_status = {}
        
        # Initialize all components
        self.embedding_provider = self._initialize_embedding_provider()
        self.search_provider = FAISSSearchProvider(self.embedding_provider, config)
        self.reranker = AdvancedReranker(config, self.embedding_provider)
        self.query_expander = QueryExpander(config, self.embedding_provider)
        self.image_processor = ImageProcessor(config)
        self.hallucination_detector = HallucinationDetector(config, self.embedding_provider)
        self.reference_manager = ReferenceDatasetManager(config)
        self.reference_evaluator = ReferenceBasedEvaluator(self.reference_manager, self.embedding_provider)
        self.feedback_manager = UserFeedbackManager(config)
        self.degraded_manager = DegradedModeManager(config)
        self.llm_provider = LLMProvider(config)
        

        
        
        self.evaluator = ProductionEvaluationProvider(self.embedding_provider, config)
        logger.info("Enhanced SwimmingRAGSystem initialized")
    
    def _initialize_embedding_provider(self) -> EmbeddingProvider:
        """Initialize embedding provider with status tracking"""
        try:
            provider = SentenceTransformerProvider(self.config.embedding_model_name)
            if provider.is_available():
                self.component_status['embedding_provider'] = True
                return provider
        except Exception as e:
            logger.warning(f"SentenceTransformer failed: {e}")
        
        self.component_status['embedding_provider'] = False
        return FallbackEmbeddingProvider(self.config.embedding_dimension)
    
    def load_data(self) -> bool:
        """Load data with component status tracking"""
        try:
             # Initialize component_status if not already done
            if not hasattr(self, 'component_status'):
                self.component_status = {}
            # Load chunks
            chunks_path = self._find_data_file(self.config.chunks_file)
            if not chunks_path:
                logger.error("Chunks file not found")
                self.component_status['data_files'] = False
                return False
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Load embeddings if available
            embeddings_path = self._find_data_file(self.config.embeddings_file)
            embeddings = None
            
            if embeddings_path:
                try:
                    with open(embeddings_path, 'rb') as f:
                        data = pickle.load(f)
                        embeddings = data['embeddings']
                except Exception as e:
                    logger.warning(f"Failed to load embeddings: {e}")
            
            # Initialize search
            self.search_provider.load_data(chunks, embeddings)
            self.component_status['search_provider'] = self.search_provider.is_ready()
            self.component_status['data_files'] = True
            self.component_status['llm_provider'] = True
            
            # Determine system mode
            self.system_mode = self.degraded_manager.determine_appropriate_mode(
                "none", self.component_status
            )
            
            if self.system_mode in [SystemMode.FULL, SystemMode.ADVANCED]:
                self.system_ready = True
                logger.info(f"System operational in {self.system_mode.value} mode")
            else:
                logger.warning(f"System running in {self.system_mode.value} mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.system_mode = SystemMode.ERROR
            if not hasattr(self, 'component_status'):
                self.component_status = {}
                return False
    
    def _find_data_file(self, filename: str) -> Optional[str]:
        """Find data file in multiple directories"""
        search_dirs = ["output1", "output", "data", ".", "files"]
        
        for directory in search_dirs:
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                return file_path
        return None
    
    def process_query(self, query: str, session_id: str = None, 
                     user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced query processing with all improvements"""
        
        start_time = time.time()
        session_id = session_id or hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            # Validate query
            if not self._validate_query(query):
                return self._create_error_response("Please ask a swimming-related question with at least 3 words.")
            
            # Check if system is in degraded mode
            if self.system_mode in [SystemMode.OFFLINE, SystemMode.MINIMAL, SystemMode.BASIC]:
                response = self.degraded_manager.get_degraded_response(query, self.system_mode)
                response['response_time'] = time.time() - start_time
                return response
            
            # Query expansion
            expanded_query = self.query_expander.expand_query(query, user_context)
            
            # Search and retrieval
            search_results = self.search_provider.search(expanded_query, self.config.rerank_top_k)
            
            # Reranking if enabled
            if self.config.enable_reranking and len(search_results) > 1:
                search_results = self.reranker.rerank_results(expanded_query, search_results)
            
            # Limit to top k
            search_results = search_results[:self.config.top_k_retrieval]
            
            # Generate response
            prompt = self.llm_provider.build_prompt(expanded_query, search_results)
            llm_response = self.llm_provider.generate_response(prompt)
            
            # Hallucination detection
            hallucination_risk = 0.0
            hallucination_details = {}
            if self.config.enable_hallucination_detection:
                hallucination_risk, hallucination_details = self.hallucination_detector.detect_hallucinations(
                    llm_response, search_results, query
                )
            
            # Process images
            images = self.image_processor.extract_images_from_results(search_results)
            
            # Format sources
            sources = self._format_sources(search_results)
            
            # Comprehensive evaluation
            evaluation = self._comprehensive_evaluation(
                query, llm_response, search_results, images, 
                hallucination_risk, hallucination_details
            )
            
            # Record interaction
            if self.config.enable_user_feedback:
                self.feedback_manager.record_interaction(session_id, "query", {
                    "query": query, 
                    "expanded_query": expanded_query,
                    "results_count": len(search_results),
                    "images_count": len(images)
                })
            
            response_time = time.time() - start_time
            
            response = {
                'text': llm_response,
                'images': images,
                'sources': sources,
                'search_results': search_results,
                'evaluation': evaluation,
                'system_status': self.system_mode.value,
                'response_time': response_time,
                'query_expanded': expanded_query != query,
                'expanded_query': expanded_query if expanded_query != query else None,
                'hallucination_risk': hallucination_risk,
                'hallucination_details': hallucination_details,
                'session_id': session_id
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return self._create_error_response(f"Processing error: {str(e)[:100]}")
        
    def process_query_with_llm_settings(self, query: str, session_id: str = None, 
                                  user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process query avec paramètres LLM personnalisés"""
    
        start_time = time.time()
        session_id = session_id or hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:8]
    
        try:
            # Configuration LLM depuis la session
            llm_enabled = st.session_state.get('enable_llm_descriptions', True)
            force_regen = st.session_state.get('force_llm_regenerate', False)
            quality_level = st.session_state.get('llm_description_quality', 'Standard')
        
            # Ajuster les paramètres du générateur selon la qualité
            if hasattr(self.image_processor, 'description_generator'):
                generator = self.image_processor.description_generator
            
                if force_regen:
                    generator.cache.clear()
            
                # Modifier le prompt selon le niveau de qualité
                if quality_level == "Détaillée":
                    generator.max_length = 200
                    generator.detail_level = "high"
                elif quality_level == "Technique":
                    generator.max_length = 150
                    generator.detail_level = "technical"
                else:
                    generator.max_length = 100
                    generator.detail_level = "standard"
        
            # Traitement normal de la requête
            if not self._validate_query(query):
                return self._create_error_response("Veuillez poser une question sur la natation avec au moins 3 mots.")
        
            if self.system_mode in [SystemMode.OFFLINE, SystemMode.MINIMAL, SystemMode.BASIC]:
                response = self.degraded_manager.get_degraded_response(query, self.system_mode)
                response['response_time'] = time.time() - start_time
                return response
        
            # Recherche et récupération
            expanded_query = self.query_expander.expand_query(query, user_context)
            search_results = self.search_provider.search(expanded_query, self.config.rerank_top_k)
        
            if self.config.enable_reranking and len(search_results) > 1:
                search_results = self.reranker.rerank_results(expanded_query, search_results)
        
            search_results = search_results[:self.config.top_k_retrieval]
        
            # Génération LLM de descriptions si activée
            if llm_enabled:
                search_results = self.regenerate_image_descriptions(search_results)
        
            # Suite du traitement normal...
            prompt = self.llm_provider.build_prompt(expanded_query, search_results)
            llm_response = self.llm_provider.generate_response(prompt)
        
            # Extraction des images avec descriptions LLM
            images = self.image_processor.extract_images_from_results(search_results)
        
            # Resto du code existant...
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return self._create_error_response(f"Erreur de traitement: {str(e)[:100]}")
    
    def _comprehensive_evaluation(self, query: str, response: str, 
                                search_results: List[Dict], images: List[Dict],
                                hallucination_risk: float, 
                                hallucination_details: Dict) -> Dict:
        """Enhanced comprehensive evaluation with all metrics"""
        
        try:
            # Standard evaluations
            retrieve_score, retrieve_details = self.evaluator.evaluate_retrieval(query, search_results)
            generate_score, generate_details = self.evaluator.evaluate_generation(query, response, search_results)
            multimodal_score, multimodal_details = self.evaluator.evaluate_multimodal(query, response, images)
            
            # Reference-based evaluation
            reference_score = 50.0
            reference_details = {}
            if self.config.enable_reference_evaluation:
                reference_score, reference_details = self.reference_evaluator.evaluate_against_reference(
                    query, response, search_results
                )
            
            # Hallucination penalty
            hallucination_penalty = hallucination_risk * 30  # Max 30 point penalty
            
            # Overall score with new weightings
            overall_score = (
                retrieve_score * 0.20 +
                generate_score * 0.35 +
                multimodal_score * 0.20 +
                reference_score * 0.25
            ) - hallucination_penalty
            
            overall_score = max(min(overall_score, 100), 0)
            
            # Performance classification
            thresholds = self.config.evaluation_thresholds
            if overall_score >= thresholds['excellent']:
                performance_level = "Excellent Performance"
                performance_class = "badge-excellent"
            elif overall_score >= thresholds['good']:
                performance_level = "Good Performance"
                performance_class = "badge-good"
            elif overall_score >= thresholds['acceptable']:
                performance_level = "Acceptable Performance"
                performance_class = "badge-medium"
            else:
                performance_level = "Needs Improvement"
                performance_class = "badge-poor"
            
            return {
                'overall_score': overall_score,
                'performance_level': performance_level,
                'performance_class': performance_class,
                'retrieve_score': retrieve_score,
                'generate_score': generate_score,
                'multimodal_score': multimodal_score,
                'reference_score': reference_score,
                'hallucination_risk': hallucination_risk,
                'details': {
                    'retrieval': retrieve_details,
                    'generation': generate_details,
                    'multimodal': multimodal_details,
                    'reference': reference_details,
                    'hallucination': hallucination_details
                },
                'recommendations': self._generate_enhanced_recommendations(
                    retrieve_score, generate_score, multimodal_score, 
                    reference_score, hallucination_risk
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return self._get_fallback_evaluation()
    
    def _generate_enhanced_recommendations(self, retrieve_score: float, 
                                         generate_score: float, 
                                         multimodal_score: float,
                                         reference_score: float,
                                         hallucination_risk: float) -> List[str]:
        """Generate enhanced recommendations"""
        recommendations = []
        
        if hallucination_risk > 0.5:
            recommendations.append("High hallucination risk detected - verify information accuracy")
        
        if retrieve_score < 50:
            recommendations.append("Document retrieval needs improvement - consider query refinement")
        elif retrieve_score >= 75:
            recommendations.append("Excellent document retrieval performance")
        
        if generate_score < 50:
            recommendations.append("Response generation could be enhanced with better context utilization")
        elif generate_score >= 75:
            recommendations.append("High-quality response generation achieved")
        
        if multimodal_score < 50:
            recommendations.append("Image-text coherence needs improvement")
        elif multimodal_score >= 75:
            recommendations.append("Excellent multimodal integration")
        
        if reference_score < 50:
            recommendations.append("Response deviates from expected reference patterns")
        elif reference_score >= 75:
            recommendations.append("Response aligns well with reference knowledge")
        
        if not recommendations:
            recommendations.append("Balanced performance across all evaluation metrics")
        
        return recommendations
    
    def _validate_query(self, query: str) -> bool:
        """Enhanced query validation"""
        query_clean = query.strip()
        
        if not query_clean or len(query_clean.split()) < 2:
            return False
        
        # Swimming context validation
        swimming_keywords = [
            'swim', 'stroke', 'technique', 'freestyle', 'backstroke', 'breaststroke', 'butterfly',
            'pool', 'water', 'training', 'drill', 'breathing', 'kick', 'pull', 'form',
            'improve', 'better', 'learn', 'practice', 'exercise', 'performance', 'coach',
            'endurance', 'speed', 'turn', 'dive', 'lap', 'race', 'competitive'
        ]
        
        return (any(keyword in query_clean.lower() for keyword in swimming_keywords) or 
                len(query_clean) > 20)  # Accept longer queries even without keywords
    
    def _format_sources(self, search_results: List[Dict]) -> List[str]:
        """Enhanced source formatting"""
        sources = []
        for i, result in enumerate(search_results[:4], 1):
            source_name = result.get('pdf', 'Swimming Guide').replace('.pdf', '').replace('_', ' ')
            stroke = result.get('stroke', 'General')
            similarity = result.get('similarity_score', 0)
            rerank_score = result.get('rerank_score', 0)
            
            source_info = f"{i}. {source_name} - {stroke}"
            if rerank_score > 0:
                source_info += f" (reranked: {rerank_score:.2f})"
            else:
                source_info += f" (similarity: {similarity:.2f})"
            
            sources.append(source_info)
        
        return sources
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'text': f"I apologize, but {message}. Please try rephrasing your swimming question.",
            'images': [],
            'sources': [],
            'search_results': [],
            'evaluation': None,
            'system_status': SystemMode.ERROR.value,  # Changed from SystemStatus to SystemMode
            'response_time': 0.0,
            'session_id': None
        }
    
    def _get_fallback_evaluation(self) -> Dict:
        """Fallback evaluation in case of errors"""
        return {
            'overall_score': 40.0,
            'performance_level': "System Error",
            'performance_class': "badge-medium",
            'retrieve_score': 40.0,
            'generate_score': 40.0,
            'multimodal_score': 40.0,
            'reference_score': 40.0,
            'hallucination_risk': 0.5,
            'details': {},
            'recommendations': ["System running in fallback mode"],
            'timestamp': datetime.now().isoformat()
        }
    
    def submit_feedback(self, session_id: str, question: str, response_data: Dict,
                       rating: int, helpfulness: int, accuracy: int, 
                       completeness: int, feedback_text: str = "") -> bool:
        """Submit user feedback"""
        try:
            if self.config.enable_user_feedback:
                self.feedback_manager.record_feedback(
                    session_id, question, response_data, rating, 
                    helpfulness, accuracy, completeness, feedback_text
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Feedback submission error: {e}")
            return False
    
    def get_system_analytics(self) -> Dict:
        """Get comprehensive system analytics"""
        try:
            if not hasattr(self, 'component_status'):
                 self.component_status = {}
            analytics = {
                'system_mode': self.system_mode.value,
                'component_status': self.component_status,
                'system_ready': self.system_ready
            }
            
            if self.config.enable_user_feedback:
                try:
                    feedback_analytics = self.feedback_manager.get_feedback_analytics()
                    analytics['feedback'] = feedback_analytics
                except Exception as e:
                    logger.warning(f"Feedback analytics error: {e}")
                    analytics['feedback'] = {'total_feedback': 0, 'avg_rating': 0}
            
            return analytics
            
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {
                'system_mode': 'error', 
                'component_status': getattr(self, 'component_status', {}),
                'system_ready': False}

# ==================== ENHANCED UI COMPONENTS ====================

class EnhancedUIComponents:
    """Enhanced UI components with new features"""
    @staticmethod
    def create_chat_entry(user_input, answer, settings):
        return {
            "question": user_input,
            "answer": answer,
            "settings": settings
        }
    
    @staticmethod
    def render_enhanced_header():
        """Enhanced header with system mode indicator"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; 
                    margin-bottom: 2rem; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);">
            <h1>🏊‍♂️ Swimming Coach AI - Advanced RAG System</h1>
            <p>Expert swimming assistance with comprehensive evaluation and feedback system</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    ✨ Reference-Based Evaluation
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    🔍 Hallucination Detection
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    📊 User Feedback System
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px;">
                    🎯 Advanced Reranking
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_enhanced_sidebar(system_mode: SystemMode, analytics: Dict):
        """Enhanced sidebar avec contrôles LLM"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration Système")
        
            # Indicateur de mode système (garder existant)
            mode_colors = {
                SystemMode.FULL: "#4caf50",
                SystemMode.ADVANCED: "#2196f3", 
                SystemMode.BASIC: "#ff9800",
                SystemMode.MINIMAL: "#ffc107",
                SystemMode.OFFLINE: "#f44336",
                SystemMode.ERROR: "#f44336"
         }
        
            mode_color = mode_colors.get(system_mode, "#9e9e9e")
            st.markdown(f"""
            <div style="background: {mode_color}; color: white; padding: 1rem; 
                        border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <strong>Mode: {system_mode.value.upper()}</strong>
            </div>
            """, unsafe_allow_html=True)
        
            # Configuration options existantes
            st.session_state.show_evaluation = st.checkbox(
                "📊 Évaluation Avancée", 
                value=st.session_state.get('show_evaluation', True)
            )
        
            st.session_state.show_details = st.checkbox(
                "🔍 Détails Techniques", 
                value=st.session_state.get('show_details', False)
            )
        
            st.session_state.show_images = st.checkbox(
                "🖼️ Contenu Visuel", 
                value=st.session_state.get('show_images', True)
            )
        
            # NOUVEAUX contrôles LLM
            st.markdown("---")
            st.markdown("### 🤖 Intelligence Artificielle")
        
            st.session_state.enable_llm_descriptions = st.checkbox(
                "Descriptions IA", 
                value=st.session_state.get('enable_llm_descriptions', True),
                help="Utiliser l'IA pour générer des descriptions d'images intelligentes"
            )
        
            if st.session_state.enable_llm_descriptions:
                st.session_state.llm_description_quality = st.selectbox(
                    "Qualité descriptions IA",
                     ["Standard", "Détaillée", "Technique"],
                    index=0,
                    help="Niveau de détail des descriptions générées"
                )
            
                st.session_state.force_llm_regenerate = st.checkbox(
                    "Forcer régénération",
                     value=False,
                    help="Régénérer toutes les descriptions même si elles existent"
                )
            
                # Statistiques du cache LLM
                if hasattr(st.session_state, 'enhanced_rag_system'):
                    rag_system = st.session_state.enhanced_rag_system
                    if (hasattr(rag_system.image_processor, 'description_generator')):
                        cache_size = len(rag_system.image_processor.description_generator.cache)
                        st.metric("Cache IA", f"{cache_size} desc.")
                    
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Vider cache", help="Effacer le cache des descriptions"):
                                rag_system.image_processor.description_generator.cache.clear()
                                st.success("Cache vidé")
                                st.rerun()
                    
                        with col2:
                            if st.button("📊 Stats IA", help="Voir les statistiques de génération"):
                                st.session_state.show_llm_stats = True
        
            #    Affichage des statistiques LLM si demandé
            if st.session_state.get('show_llm_stats', False):
                st.markdown("#### 🤖 Statistiques IA")
                if hasattr(st.session_state, 'enhanced_rag_system'):
                    rag_system = st.session_state.enhanced_rag_system
                    if hasattr(rag_system.image_processor, 'description_generator'):
                        generator = rag_system.image_processor.description_generator
                    
                        # Analyser le cache
                        cache_stats = {}
                        for key, desc in generator.cache.items():
                            length = len(desc)
                            if length < 50:
                               category = "Courte"
                            elif length < 100:
                                category = "Moyenne" 
                            else:
                                category = "Détaillée"
                        
                            cache_stats[category] = cache_stats.get(category, 0) + 1
                    
                        for category, count in cache_stats.items():
                            st.metric(f"Desc. {category.lower()}s", count)
            
                if st.button("❌ Fermer stats"):
                    st.session_state.show_llm_stats = False
                    st.rerun()

    
    @staticmethod
    def render_feedback_interface(session_id: str, question: str, 
                                response_data: Dict, rag_system, chat_id) -> bool:
        
        """Render user feedback interface"""
        rating = st.slider("Overall Rating", 1, 5, 3, key=f"rating_{session_id}_{chat_id}")
        if not st.session_state.get('enable_feedback', True):
            return False
        
        with st.expander("📝 Rate This Response", expanded=False):
            st.markdown("**Help us improve by rating this response:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                helpfulness = st.slider("Helpfulness", 1, 5, 3, key=f"help_{session_id}_{chat_id}_{hash(question)}")
            
            with col2:
                accuracy = st.slider("Accuracy", 1, 5, 3,  key=f"acc_{session_id}_{chat_id}_{hash(question)}")
                
                completeness = st.slider("Completeness", 1, 5, 3, key=f"comp_{session_id}_{chat_id}_{hash(question)}")
            
            feedback_text = st.text_area(
                "Additional Comments (Optional)", 
                placeholder="What could be improved?",
                 key=f"feedback_{session_id}_{chat_id}_{hash(question)}"
            )
            
            if st.button("Submit Feedback", key=f"submit_{session_id}_{chat_id}_{hash(question)}"):
                success = rag_system.submit_feedback(
                    session_id, question, response_data, rating, 
                    helpfulness, accuracy, completeness, feedback_text
                )
                
                if success:
                    st.success("Thank you for your feedback!")
                    return True
                else:
                    st.error("Failed to submit feedback. Please try again.")
        
        return False
    
    @staticmethod
    def render_enhanced_evaluation(evaluation: Dict):
        """Enhanced evaluation display with new metrics"""
        if not evaluation:
            return
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.95); border-radius: 12px; 
                    padding: 1.5rem; margin: 1rem 0; box-shadow: 0 6px 12px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3>📊 Enhanced RAG Evaluation</h3>
                <div style="background: {UIComponents._get_badge_color(evaluation['performance_class'])}; 
                           color: white; padding: 0.5rem 1rem; border-radius: 20px; display: inline-block;">
                    {evaluation['performance_level']} - {evaluation['overall_score']:.1f}/100
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced metrics grid
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
                        gap: 1rem; margin: 1rem 0;">
                <div style="background: white; border-radius: 10px; padding: 1rem; text-align: center; 
                           box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-left: 3px solid {UIComponents._get_score_color(evaluation['retrieve_score'])};">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">🔍 Retrieval</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #333;">{evaluation['retrieve_score']:.1f}</div>
                </div>
                <div style="background: white; border-radius: 10px; padding: 1rem; text-align: center; 
                           box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-left: 3px solid {UIComponents._get_score_color(evaluation['generate_score'])};">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">✍️ Generation</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #333;">{evaluation['generate_score']:.1f}</div>
                </div>
                <div style="background: white; border-radius: 10px; padding: 1rem; text-align: center; 
                           box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-left: 3px solid {UIComponents._get_score_color(evaluation['multimodal_score'])};">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">🖼️ Multimodal</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #333;">{evaluation['multimodal_score']:.1f}</div>
                </div>
                <div style="background: white; border-radius: 10px; padding: 1rem; text-align: center; 
                           box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-left: 3px solid {UIComponents._get_score_color(evaluation['reference_score'])};">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">📚 Reference</div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #333;">{evaluation['reference_score']:.1f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Hallucination risk indicator
        if evaluation.get('hallucination_risk', 0) > 0.3:
            risk_color = "#f44336" if evaluation['hallucination_risk'] > 0.7 else "#ff9800"
            st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 0.5rem 1rem; 
                        border-radius: 8px; text-align: center; margin: 0.5rem 0;">
                ⚠️ Hallucination Risk: {evaluation['hallucination_risk']:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced recommendations
        if evaluation.get('recommendations'):
            st.markdown("**💡 System Recommendations:**")
            for rec in evaluation['recommendations']:
                st.markdown(f"- {rec}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def render_chat_entry_enhanced(chat: Dict, image_processor, rag_system):
        """Enhanced chat entry rendering with feedback"""
        # Question
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 15px; margin: 0.5rem 0; 
                    background: linear-gradient(135deg, #4fc3f7 0%, #0288d1 100%); 
                    color: white; margin-left: 2rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <strong>🏊‍♀️ You ({chat['timestamp']}):</strong><br>
            {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Response with enhanced status
        response_text = chat['response'].get('text', 'No response generated')
        system_status = chat['response'].get('system_status', 'unknown')
        response_time = chat['response'].get('response_time', 0)
        
        status_indicators = {
            'full': '🏊‍♂️ Expert Coach (Full System)',
            'advanced': '🏊‍♂️ Coach (Advanced Mode)',
            'basic': '🏊‍♂️ Coach (Basic Mode)',
            'minimal': '⚠️ Coach (Minimal Mode)',
            'offline': '📱 Coach (Offline Mode)',
            'error': '⚠️ Coach (Limited)'
        }
        
        status_indicator = status_indicators.get(system_status, '🏊‍♂️ Coach')
        
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 15px; margin: 0.5rem 0; 
                    background: linear-gradient(135deg, #26a69a 0%, #00796b 100%); 
                    color: white; margin-right: 2rem; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <strong>{status_indicator}</strong>
            <span style="float: right; font-size: 0.8rem;">⏱️ {response_time:.1f}s</span><br>
            {response_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Query expansion indicator
        if chat['response'].get('query_expanded'):
            st.info(f"🔍 Query expanded to: {chat['response'].get('expanded_query')}")
        
        # Enhanced evaluation
        if chat['settings']['show_evaluation'] and chat['response'].get('evaluation'):
            EnhancedUIComponents.render_enhanced_evaluation(chat['response']['evaluation'])
        
        # User feedback interface
        session_id = chat['response'].get('session_id')
        if session_id and st.session_state.get('enable_feedback', True):
            chat_id = chat.get('id', str(uuid.uuid4()))  # génère un id unique si inexistant
            EnhancedUIComponents.render_feedback_interface(
                session_id, chat['question'], chat['response'], rag_system, chat_id
                )
        
        # Images and other components (same as before)
        if chat['settings']['show_images'] and chat['response'].get('images'):
            UIComponents._render_images(chat['response']['images'],image_processor=image_processor,chat_id=chat_id, question=chat['question'])
        
        if chat['response'].get('sources'):
            UIComponents._render_sources(chat['response']['sources'])
        
        if chat['settings']['show_details'] and chat['response'].get('search_results'):
            UIComponents._render_technical_details(chat['response']['search_results'])
    import uuid  # mets ça en haut de ton fichier


# ==================== PRODUCTION EVALUATOR ====================

class ProductionEvaluationProvider(EvaluationProvider):
    """Enhanced production evaluator with all improvements"""
    
    def __init__(self, embedding_provider: EmbeddingProvider, config: RAGConfig):
        self.embedding_provider = embedding_provider
        self.config = config
        
        # Enhanced swimming vocabulary
        self.swimming_vocabulary = {
            'strokes': ['freestyle', 'backstroke', 'breaststroke', 'butterfly', 'crawl'],
            'techniques': ['technique', 'form', 'breathing', 'kick', 'pull', 'catch', 'stroke'],
            'training': ['drill', 'exercise', 'practice', 'training', 'workout', 'set'],
            'performance': ['speed', 'endurance', 'efficiency', 'improvement', 'time'],
            'biomechanics': ['rotation', 'alignment', 'streamline', 'coordination', 'timing'],
            'equipment': ['goggles', 'kickboard', 'pull buoy', 'fins', 'paddles']
        }
        
        # Initialize readability analyzer
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.nltk_available = True
        except:
            self.nltk_available = False
            logger.warning("NLTK not available for advanced text analysis")
    
    def evaluate_retrieval(self, query: str, results: List[Dict]) -> Tuple[float, Dict]:
        """Enhanced retrieval evaluation"""
        details = {
            "num_results": len(results),
            "semantic_similarity": 0.0,
            "domain_relevance": 0.0,
            "content_diversity": 0.0,
            "technical_depth": 0.0,
            "source_quality": 0.0,
            "reranking_effectiveness": 0.0
        }
        
        if not results:
            return 25.0, details
        
        try:
            # Enhanced metrics
            details["semantic_similarity"] = self._evaluate_semantic_similarity_enhanced(query, results)
            details["domain_relevance"] = self._evaluate_domain_relevance_enhanced(results)
            details["content_diversity"] = self._evaluate_content_diversity_enhanced(results)
            details["technical_depth"] = self._evaluate_technical_depth_enhanced(results)
            details["source_quality"] = self._evaluate_source_quality(results)
            details["reranking_effectiveness"] = self._evaluate_reranking_effectiveness(results)
            
            # Weighted score
            retrieve_score = (
                details["semantic_similarity"] * 0.25 +
                details["domain_relevance"] * 0.20 +
                details["technical_depth"] * 0.20 +
                details["content_diversity"] * 0.15 +
                details["source_quality"] * 0.10 +
                details["reranking_effectiveness"] * 0.10
            )
            
            return min(max(retrieve_score, 0), 100), details
            
        except Exception as e:
            logger.error(f"Enhanced retrieval evaluation error: {e}")
            return 40.0, details
    
    def evaluate_generation(self, query: str, response: str, context: List[Dict]) -> Tuple[float, Dict]:
        """Enhanced generation evaluation"""
        details = {
            "response_length": len(response),
            "query_coverage": 0.0,
            "context_faithfulness": 0.0,
            "technical_accuracy": 0.0,
            "actionability": 0.0,
            "readability": 0.0,
            "coherence": 0.0,
            "safety": 0.0
        }
        
        if not response or len(response) < 10:
            return 20.0, details
        
        try:
            response_lower = response.lower()
            
            # Enhanced metrics
            details["query_coverage"] = self._evaluate_query_coverage_enhanced(query, response)
            details["context_faithfulness"] = self._evaluate_context_faithfulness_enhanced(response, context)
            details["technical_accuracy"] = self._evaluate_technical_accuracy_enhanced(response_lower)
            details["actionability"] = self._evaluate_actionability_enhanced(response_lower)
            details["readability"] = self._evaluate_readability(response)
            details["coherence"] = self._evaluate_coherence(response)
            details["safety"] = self._evaluate_safety(response_lower)
            
            # Weighted score
            generate_score = (
                details["query_coverage"] * 0.25 +
                details["context_faithfulness"] * 0.20 +
                details["technical_accuracy"] * 0.20 +
                details["actionability"] * 0.15 +
                details["readability"] * 0.10 +
                details["coherence"] * 0.05 +
                details["safety"] * 0.05
            )
            
            return min(max(generate_score, 0), 100), details
            
        except Exception as e:
            logger.error(f"Enhanced generation evaluation error: {e}")
            return 45.0, details
    
    def evaluate_multimodal(self, query: str, response: str, images: List[Dict]) -> Tuple[float, Dict]:
        """Enhanced multimodal evaluation"""
        details = {
            "num_images": len(images),
            "image_query_relevance": 0.0,
            "image_response_coherence": 0.0,
            "description_quality": 0.0,
            "visual_instruction_value": 0.0,
            "image_diversity": 0.0,
            "technical_demonstration": 0.0
        }
        
        if not images:
            return 35.0, details
        
        try:
            # Enhanced multimodal metrics
            details["image_query_relevance"] = self._evaluate_image_query_relevance_enhanced(query, images)
            details["image_response_coherence"] = self._evaluate_image_response_coherence_enhanced(response, images)
            details["description_quality"] = self._evaluate_description_quality_enhanced(images)
            details["visual_instruction_value"] = self._evaluate_visual_instruction_value_enhanced(images)
            details["image_diversity"] = self._evaluate_image_diversity(images)
            details["technical_demonstration"] = self._evaluate_technical_demonstration(images, query)
            
            # Weighted score
            base_score = (
                details["image_query_relevance"] * 0.25 +
                details["image_response_coherence"] * 0.20 +
                details["description_quality"] * 0.20 +
                details["visual_instruction_value"] * 0.15 +
                details["image_diversity"] * 0.10 +
                details["technical_demonstration"] * 0.10
            )
            
            # Optimal number bonus/penalty
            if 2 <= len(images) <= 4:
                base_score += 5
            elif len(images) > 6:
                base_score -= 3
            
            return min(max(base_score, 0), 100), details
            
        except Exception as e:
            logger.error(f"Enhanced multimodal evaluation error: {e}")
            return 40.0, details
    
    # Enhanced helper methods with improved algorithms
    
    def _evaluate_semantic_similarity_enhanced(self, query: str, results: List[Dict]) -> float:
        """Enhanced semantic similarity evaluation"""
        try:
            if self.embedding_provider.is_available():
                query_embedding = self.embedding_provider.encode([query])
                result_texts = [result.get('text', '')[:400] for result in results]
                result_embeddings = self.embedding_provider.encode(result_texts)
                
                similarities = cosine_similarity(query_embedding, result_embeddings)[0]
                
                # Weight by position (earlier results more important)
                weights = [1.0 / (i + 1) for i in range(len(similarities))]
                weighted_avg = np.average(similarities, weights=weights)
                
                return max(float(weighted_avg) * 100, 30)
            else:
                return self._keyword_similarity_enhanced(query, results)
                
        except Exception as e:
            logger.warning(f"Enhanced semantic similarity error: {e}")
            return self._keyword_similarity_enhanced(query, results)
    
    def _keyword_similarity_enhanced(self, query: str, results: List[Dict]) -> float:
        """Enhanced keyword-based similarity"""
        query_words = set(query.lower().split())
        similarities = []
        
        for result in results:
            text_words = set(result.get('text', '').lower().split())
            if query_words and text_words:
                # Jaccard similarity
                intersection = len(query_words.intersection(text_words))
                union = len(query_words.union(text_words))
                jaccard = intersection / union if union > 0 else 0
                
                # Boost for swimming domain terms
                domain_boost = sum(0.1 for category in self.swimming_vocabulary.values()
                                 for term in category if term in text_words) 
                
                total_score = (jaccard + domain_boost) * 100
                similarities.append(min(total_score, 100))
        
        return max(np.mean(similarities) if similarities else 30, 30)
    
    def _evaluate_domain_relevance_enhanced(self, results: List[Dict]) -> float:
        """Enhanced domain relevance evaluation"""
        scores = []
        
        for result in results:
            text_lower = result.get('text', '').lower()
            domain_score = 0
            
            # Count domain-specific terms with weights
            for category, terms in self.swimming_vocabulary.items():
                category_weight = {
                    'strokes': 1.0,
                    'techniques': 0.9,
                    'training': 0.8,
                    'performance': 0.7,
                    'biomechanics': 0.9,
                    'equipment': 0.6
                }.get(category, 0.5)
                
                term_count = sum(1 for term in terms if term in text_lower)
                domain_score += term_count * category_weight
            
            # Normalize by text length and add baseline
            text_length = len(text_lower.split())
            if text_length > 0:
                density_score = (domain_score / text_length) * 500  # Scale factor
                final_score = min(density_score + 35, 100)
            else:
                final_score = 35
            
            scores.append(final_score)
        
        return float(np.mean(scores)) if scores else 35
    
    def _evaluate_content_diversity_enhanced(self, results: List[Dict]) -> float:
        """Enhanced content diversity evaluation"""
        # Source diversity
        sources = [result.get('pdf', 'unknown') for result in results]
        unique_sources = len(set(sources))
        source_diversity = min(unique_sources * 15 + 20, 70)
        
        # Type diversity
        types = [result.get('swimming_type', 'general') for result in results]
        unique_types = len(set(types))
        type_diversity = min(unique_types * 12 + 15, 50)
        
        # Stroke diversity
        strokes = [result.get('stroke', 'general') for result in results]
        unique_strokes = len(set(strokes))
        stroke_diversity = min(unique_strokes * 8 + 10, 35)
        
        # Length diversity (variety in content length)
        lengths = [len(result.get('text', '')) for result in results]
        if len(set(lengths)) > 1:
            length_diversity = min(np.std(lengths) / 100, 15)
        else:
            length_diversity = 5
        
        total_diversity = (
            source_diversity * 0.4 + 
            type_diversity * 0.3 + 
            stroke_diversity * 0.2 + 
            length_diversity * 0.1
        )
        
        return min(total_diversity, 100)
    
    def _evaluate_technical_depth_enhanced(self, results: List[Dict]) -> float:
        """Enhanced technical depth evaluation"""
        scores = []
        
        technical_indicators = {
            'basic': ['technique', 'form', 'position', 'breathing', 'kick'],
            'intermediate': ['coordination', 'timing', 'efficiency', 'stroke rate', 'rhythm'],
            'advanced': ['biomechanics', 'hydrodynamics', 'periodization', 'lactate', 'vo2max']
        }
        
        for result in results:
            text_lower = result.get('text', '').lower()
            depth_score = 0
            
            # Score by technical level
            for level, indicators in technical_indicators.items():
                weight = {'basic': 1.0, 'intermediate': 1.5, 'advanced': 2.0}[level]
                count = sum(1 for indicator in indicators if indicator in text_lower)
                depth_score += count * weight * 5
            
            # Bonus for numerical data, measurements, specific instructions
            if re.search(r'\d+\s*(seconds?|minutes?|meters?|%|times?)', text_lower):
                depth_score += 10
            
            if any(word in text_lower for word in ['research', 'study', 'analysis']):
                depth_score += 8
            
            # Content length bonus (more detailed explanations)
            text_length = len(result.get('text', ''))
            length_bonus = min(text_length / 100, 15)
            
            final_score = min(depth_score + length_bonus + 25, 100)
            scores.append(final_score)
        
        return float(np.mean(scores)) if scores else 35
    
    def _evaluate_source_quality(self, results: List[Dict]) -> float:
        """Evaluate quality of sources"""
        quality_scores = []
        
        for result in results:
            quality_score = 50  # Base score
            
            # PDF source quality indicators
            pdf_name = result.get('pdf', '').lower()
            if 'coaching' in pdf_name or 'technique' in pdf_name:
                quality_score += 15
            if 'guide' in pdf_name or 'manual' in pdf_name:
                quality_score += 10
            
            # Content quality indicators
            text = result.get('text', '')
            if len(text) > 200:  # Substantial content
                quality_score += 10
            if len(text) > 500:  # Comprehensive content
                quality_score += 10
            
            # Structure indicators
            if '.' in text and ':' in text:  # Well-structured
                quality_score += 8
            
            # Similarity score indicates relevance
            sim_score = result.get('similarity_score', 0)
            quality_score += min(sim_score * 20, 15)
            
            quality_scores.append(min(quality_score, 100))
        
        return float(np.mean(quality_scores)) if quality_scores else 50
    
    def _evaluate_reranking_effectiveness(self, results: List[Dict]) -> float:
        """Evaluate effectiveness of reranking"""
        rerank_scores = [result.get('rerank_score', 0) for result in results]
        
        if not any(score > 0 for score in rerank_scores):
            return 50  # No reranking applied
        
        # Check if reranking improved ordering
        original_scores = [result.get('similarity_score', 0) for result in results]
        
        # Simple effectiveness measure: correlation with position
        position_weights = [1.0 / (i + 1) for i in range(len(rerank_scores))]
        
        if rerank_scores:
            weighted_score = np.average(rerank_scores, weights=position_weights)
            effectiveness = min(weighted_score * 100, 100)
        else:
            effectiveness = 50
        
        return effectiveness
    
    def _evaluate_query_coverage_enhanced(self, query: str, response: str) -> float:
        """Enhanced query coverage evaluation"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 50
        
        # Direct coverage
        direct_coverage = len(query_words.intersection(response_words)) / len(query_words)
        
        # Semantic coverage using swimming synonyms
        semantic_matches = 0
        swimming_synonyms = {
            'freestyle': ['crawl', 'front crawl'],
            'improve': ['better', 'enhance', 'develop'],
            'technique': ['form', 'mechanics', 'method'],
            'breathing': ['breath', 'air', 'ventilation'],
            'speed': ['fast', 'quick', 'velocity'],
            'training': ['practice', 'drill', 'workout']
        }
        
        for query_word in query_words:
            if query_word not in response_words:
                if query_word in swimming_synonyms:
                    synonyms = swimming_synonyms[query_word]
                    if any(syn in response_words for syn in synonyms):
                        semantic_matches += 1
        
        semantic_coverage = semantic_matches / len(query_words) if query_words else 0
        
        # Intent coverage (question words, action words)
        intent_words = {'how', 'what', 'why', 'when', 'improve', 'learn', 'fix'}
        intent_in_query = query_words.intersection(intent_words)
        
        intent_coverage = 0
        if intent_in_query:
            # Check if response addresses the intent
            action_words = {'practice', 'focus', 'try', 'work', 'train', 'maintain'}
            if response_words.intersection(action_words):
                intent_coverage = 0.3
        
        total_coverage = (direct_coverage * 0.6 + semantic_coverage * 0.3 + intent_coverage * 0.1) * 100
        return min(max(total_coverage, 25), 100)
    
    def _evaluate_context_faithfulness_enhanced(self, response: str, context: List[Dict]) -> float:
        """Enhanced context faithfulness evaluation"""
        if not context:
            return 60  # Neutral score if no context
        
        response_words = set(response.lower().split())
        context_concepts = set()
        specific_facts = []
        
        # Extract concepts and facts from context
        for chunk in context:
            text = chunk.get('text', '').lower()
            words = text.split()
            context_concepts.update(words)
            
            # Extract specific facts (numbers, proper nouns, technical terms)
            facts = re.findall(r'\b\d+\s*(?:seconds?|minutes?|meters?|feet|%)\b', text)
            specific_facts.extend(facts)
        
        if not context_concepts:
            return 60
        
        # Concept alignment
        concept_overlap = len(response_words.intersection(context_concepts))
        concept_score = min((concept_overlap / len(response_words)) * 150, 80) if response_words else 40
        
        # Specific fact consistency
        fact_score = 70  # Base score
        response_lower = response.lower()
        
        for fact in specific_facts[:5]:  # Check first 5 facts
            if fact.lower() in response_lower:
                fact_score += 6  # Bonus for including specific facts
        
        # Citation indicators
        citation_indicators = ['according to', 'research shows', 'studies indicate', 'experts recommend']
        citation_bonus = sum(5 for indicator in citation_indicators if indicator in response_lower)
        
        # Contradiction check (simple)
        contradiction_penalty = 0
        contradiction_pairs = [('always', 'never'), ('increase', 'decrease'), ('more', 'less')]
        
        for context_chunk in context:
            context_lower = context_chunk.get('text', '').lower()
            for word1, word2 in contradiction_pairs:
                if word1 in context_lower and word2 in response_lower:
                    contradiction_penalty += 5
                elif word2 in context_lower and word1 in response_lower:
                    contradiction_penalty += 5
        
        total_score = (concept_score * 0.6 + fact_score * 0.4) + citation_bonus - contradiction_penalty
        return min(max(total_score, 20), 100)
    
    def _evaluate_technical_accuracy_enhanced(self, response_lower: str) -> float:
        """Enhanced technical accuracy evaluation"""
        accuracy_score = 40  # Base score
        
        # Count technical terms by category with weights
        for category, terms in self.swimming_vocabulary.items():
            weight = {
                'strokes': 1.0, 'techniques': 1.2, 'biomechanics': 1.5,
                'training': 0.8, 'performance': 0.9, 'equipment': 0.6
            }.get(category, 1.0)
            
            term_count = sum(1 for term in terms if term in response_lower)
            accuracy_score += term_count * 3 * weight
        
        # Precision indicators
        precision_indicators = {
            'high': ['specifically', 'precisely', 'exactly', 'optimal'],
            'medium': ['properly', 'correctly', 'effectively', 'recommended'],
            'low': ['generally', 'typically', 'usually', 'often']
        }
        
        for level, indicators in precision_indicators.items():
            weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[level]
            count = sum(1 for indicator in indicators if indicator in response_lower)
            accuracy_score += count * 4 * weight
        
        # Technical measurements and specifics
        if re.search(r'\d+\s*(?:seconds?|minutes?|meters?|times?|%)', response_lower):
            accuracy_score += 8
        
        # Penalties for vague or incorrect terms
        vague_terms = ['maybe', 'perhaps', 'might', 'could be', 'sometimes']
        vagueness_penalty = sum(3 for term in vague_terms if term in response_lower)
        
        # Dangerous advice penalty
        dangerous_terms = ['hold breath', 'hyperventilate', 'push through pain']
        danger_penalty = sum(10 for term in dangerous_terms if term in response_lower)
        
        final_score = accuracy_score - vagueness_penalty - danger_penalty
        return min(max(final_score, 15), 100)
    
    def _evaluate_actionability_enhanced(self, response_lower: str) -> float:
        """Enhanced actionability evaluation"""
        actionability_score = 30  # Base score
        
        # Action words with weights
        action_categories = {
            'practice': ['practice', 'drill', 'exercise', 'train', 'work on'],
            'focus': ['focus', 'concentrate', 'emphasize', 'pay attention'],
            'maintain': ['maintain', 'keep', 'hold', 'sustain'],
            'improve': ['improve', 'develop', 'enhance', 'strengthen'],
            'avoid': ['avoid', 'don\'t', 'prevent', 'stop']
        }
        
        for category, words in action_categories.items():
            count = sum(1 for word in words if word in response_lower)
            actionability_score += count * 5
        
        # Specific instructions
        instruction_patterns = [
            r'\d+\s*times', r'\d+\s*seconds', r'\d+\s*minutes', r'\d+\s*reps',
            r'step \d+', r'first.*then', r'begin.*with', r'start.*by'
        ]
        
        instruction_count = sum(1 for pattern in instruction_patterns 
                              if re.search(pattern, response_lower))
        actionability_score += instruction_count * 8
        
        # Structure indicators
        structure_words = ['first', 'second', 'then', 'next', 'finally', 'also', 'additionally']
        structure_count = sum(1 for word in structure_words if word in response_lower)
        actionability_score += min(structure_count * 3, 15)
        
        # Progressive instructions
        if any(phrase in response_lower for phrase in ['gradually', 'start with', 'progress to', 'build up']):
            actionability_score += 10
        
        return min(actionability_score, 100)
    
    def _evaluate_readability(self, response: str) -> float:
        """Evaluate response readability"""
        try:
            # Use textstat for readability analysis
            flesch_score = flesch_reading_ease(response)
            
            # Convert Flesch score to 0-100 scale
            if flesch_score >= 80:  # Very easy
                readability = 95
            elif flesch_score >= 70:  # Easy
                readability = 85
            elif flesch_score >= 60:  # Standard
                readability = 75
            elif flesch_score >= 50:  # Fairly difficult
                readability = 65
            elif flesch_score >= 30:  # Difficult
                readability = 45
            else:  # Very difficult
                readability = 25
            
            return readability
            
        except Exception as e:
            logger.warning(f"Readability analysis error: {e}")
            
            # Fallback readability analysis
            sentences = response.split('.')
            words = response.split()
            
            if not sentences or not words:
                return 50
            
            avg_sentence_length = len(words) / len(sentences)
            
            # Simple readability heuristic
            if avg_sentence_length <= 15:
                return 80
            elif avg_sentence_length <= 20:
                return 70
            elif avg_sentence_length <= 25:
                return 60
            else:
                return 40
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate response coherence"""
        coherence_score = 60  # Base score
        
        # Transition words
        transitions = ['however', 'therefore', 'additionally', 'furthermore', 
                      'consequently', 'meanwhile', 'also', 'then', 'next']
        
        transition_count = sum(1 for transition in transitions if transition in response.lower())
        coherence_score += min(transition_count * 5, 20)
        
        # Repetitive content penalty
        words = response.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 4:  # Only count substantial words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_penalty = sum(max(0, count - 2) for count in word_counts.values()) * 2
        coherence_score -= min(repetition_penalty, 15)
        
        # Topic consistency (swimming domain)
        domain_terms = sum(1 for category in self.swimming_vocabulary.values()
                          for term in category if term in response.lower())
        
        if domain_terms >= 3:
            coherence_score += 10
        
        return min(max(coherence_score, 30), 100)
    
    def _evaluate_safety(self, response_lower: str) -> float:
        """Evaluate response safety"""
        safety_score = 90  # High base score (assume safe)
        
        # Dangerous advice patterns
        dangerous_patterns = [
            r'hold.*breath.*\d+.*minutes',
            r'swim.*alone.*ocean',
            r'ignore.*pain',
            r'hyperventilate',
            r'dive.*shallow.*water'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, response_lower):
                safety_score -= 20
        
        # Safety promoting terms
        safety_terms = ['safety', 'careful', 'supervision', 'gradually', 'listen to your body']
        safety_bonus = sum(5 for term in safety_terms if term in response_lower)
        safety_score += min(safety_bonus, 15)
        
        return min(max(safety_score, 40), 100)
    
    # Enhanced image evaluation methods
    def _evaluate_image_query_relevance_enhanced(self, query: str, images: List[Dict]) -> float:
        """Enhanced image-query relevance evaluation"""
        if not images:
            return 0
        
        query_words = set(query.lower().split())
        relevance_scores = []
        
        for img in images:
            description = img.get('enhanced_description', '').lower()
            stroke_focus = img.get('stroke_focus', '').lower()
            source = img.get('source', '').lower()
            
            all_img_words = set(description.split() + stroke_focus.split() + source.split())
            
            if query_words:
                # Direct overlap
                overlap_ratio = len(query_words.intersection(all_img_words)) / len(query_words)
                
                # Semantic relevance bonus
                semantic_bonus = 0
                if any(stroke in query.lower() for stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']):
                    if any(stroke in stroke_focus for stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']):
                        semantic_bonus = 0.2
                
                relevance_scores.append((overlap_ratio + semantic_bonus) * 100)
            else:
                relevance_scores.append(40)
        
        return max(np.mean(relevance_scores), 25) if relevance_scores else 25
    
    def _evaluate_image_response_coherence_enhanced(self, response: str, images: List[Dict]) -> float:
        """Enhanced image-response coherence evaluation"""
        response_words = set(response.lower().split())
        coherence_scores = []
        
        for img in images:
            description = img.get('enhanced_description', '').lower()
            stroke_focus = img.get('stroke_focus', '').lower()
            
            img_words = set(description.split() + stroke_focus.split())
            
            if response_words and img_words:
                # Direct coherence
                coherence = len(response_words.intersection(img_words)) / min(len(response_words), 40)
                
                # Context coherence (if response mentions the stroke that image shows)
                context_bonus = 0
                for stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']:
                    if stroke in response.lower() and stroke in stroke_focus:
                        context_bonus = 0.15
                        break
                
                total_coherence = (coherence + context_bonus) * 100
                coherence_scores.append(min(total_coherence, 100))
            else:
                coherence_scores.append(30)
        
        return max(np.mean(coherence_scores), 30) if coherence_scores else 30
    
    def _evaluate_description_quality_enhanced(self, images: List[Dict]) -> float:
        """Enhanced image description quality evaluation"""
        quality_scores = []
        
        for img in images:
            description = img.get('enhanced_description', '')
            
            if not description or len(description) < 10:
                quality_scores.append(15)
                continue
            
            quality_score = 30  # Base score
            
            # Length and detail
            word_count = len(description.split())
            if word_count >= 10:
                quality_score += 15
            if word_count >= 15:
                quality_score += 10
            
            # Technical specificity
            technical_terms = ['technique', 'demonstration', 'instruction', 'biomechanics']
            tech_count = sum(1 for term in technical_terms if term.lower() in description.lower())
            quality_score += tech_count * 5
            
            # Descriptive richness
            descriptive_words = ['professional', 'expert', 'detailed', 'comprehensive', 'focused']
            desc_count = sum(1 for word in descriptive_words if word.lower() in description.lower())
            quality_score += desc_count * 3
            
            # Swimming domain relevance
            domain_relevance = sum(1 for category in self.swimming_vocabulary.values()
                                 for term in category if term in description.lower())
            quality_score += min(domain_relevance * 2, 15)
            
            quality_scores.append(min(quality_score, 100))
        
        return np.mean(quality_scores) if quality_scores else 30
    
    def _evaluate_visual_instruction_value_enhanced(self, images: List[Dict]) -> float:
        """Enhanced visual instruction value evaluation"""
        value_scores = []
        
        for img in images:
            instruction_value = 35  # Base score
            
            description = img.get('enhanced_description', '').lower()
            similarity_score = img.get('similarity_score', 0)
            stroke_focus = img.get('stroke_focus', '').lower()
            
            # Instructional terms
            instruction_terms = ['instruction', 'demonstration', 'technique', 'training', 'guide', 'coaching']
            instruction_count = sum(1 for term in instruction_terms if term in description)
            instruction_value += instruction_count * 6
            
            # Search relevance
            relevance_bonus = min(similarity_score * 25, 20) if similarity_score else 10
            instruction_value += relevance_bonus
            
            # Stroke specificity
            if stroke_focus and stroke_focus != 'general':
                instruction_value += 12
            
            # Visual learning indicators
            visual_terms = ['form', 'position', 'movement', 'coordination', 'timing']
            visual_count = sum(1 for term in visual_terms if term in description)
            instruction_value += visual_count * 4
            
            value_scores.append(min(instruction_value, 100))
        
        return np.mean(value_scores) if value_scores else 30
    
    def _evaluate_image_diversity(self, images: List[Dict]) -> float:
        """Evaluate diversity of images"""
        if not images:
            return 0
        
        # Stroke diversity
        strokes = [img.get('stroke_focus', 'general').lower() for img in images]
        unique_strokes = len(set(strokes))
        stroke_diversity = min(unique_strokes * 20, 60)
        
        # Source diversity
        sources = [img.get('source', 'unknown') for img in images]
        unique_sources = len(set(sources))
        source_diversity = min(unique_sources * 15, 40)
        
        # Description diversity (different types of content)
        descriptions = [img.get('enhanced_description', '') for img in images]
        unique_descriptions = len(set(desc[:50] for desc in descriptions))  # First 50 chars
        desc_diversity = min(unique_descriptions * 10, 30)
        
        total_diversity = stroke_diversity + source_diversity + desc_diversity
        return min(total_diversity, 100)
    
    def _evaluate_technical_demonstration(self, images: List[Dict], query: str) -> float:
        """Evaluate how well images demonstrate technical aspects"""
        demo_scores = []
        query_lower = query.lower()
        
        # Technical aspects mentioned in query
        technical_aspects = {
            'technique': ['technique', 'form', 'mechanics'],
            'breathing': ['breathing', 'breath', 'air'],
            'stroke': ['stroke', 'arm', 'pull', 'catch'],
            'kick': ['kick', 'leg', 'flutter', 'dolphin'],
            'position': ['position', 'body', 'alignment'],
            'timing': ['timing', 'coordination', 'rhythm']
        }
        
        query_aspects = []
        for aspect, keywords in technical_aspects.items():
            if any(keyword in query_lower for keyword in keywords):
                query_aspects.append(aspect)
        
        if not query_aspects:
            return 60  # Default if no specific technical aspects identified
        
        for img in images:
            demo_score = 40  # Base score
            description = img.get('enhanced_description', '').lower()
            
            # Check if image addresses query aspects
            for aspect in query_aspects:
                if any(keyword in description for keyword in technical_aspects[aspect]):
                    demo_score += 15
            
            # Technical demonstration quality
            demo_terms = ['demonstration', 'shows', 'illustrates', 'depicts', 'displays']
            if any(term in description for term in demo_terms):
                demo_score += 10
            
            # Specific technique focus
            if 'technique' in description or 'form' in description:
                demo_score += 8
            
            demo_scores.append(min(demo_score, 100))
        
        return np.mean(demo_scores) if demo_scores else 50

# ==================== MAIN APPLICATION ====================

def main():
    """Enhanced main application"""
    
    # Streamlit configuration
    st.set_page_config(
        page_title="Swimming Coach AI - Advanced RAG",
        page_icon="🏊‍♂️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        .metric-card { 
            transition: all 0.3s ease;
            border-radius: 10px;
        }
        .metric-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .evaluation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize enhanced system
    if 'enhanced_rag_system' not in st.session_state:
        with st.spinner("🚀 Initializing Enhanced RAG System..."):
            try:
                config = RAGConfig()
                st.session_state.enhanced_rag_system = EnhancedSwimmingRAGSystem(config)
                
                # Load data
                if st.session_state.enhanced_rag_system.load_data():
                    st.success("✅ Enhanced RAG system initialized successfully!")
                    logger.info("Enhanced system ready")
                else:
                    st.warning("⚠️ System initialized in degraded mode")
                    
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                st.session_state.enhanced_rag_system = None
                logger.error(f"System init error: {e}")
    
    rag_system = st.session_state.enhanced_rag_system
    if not rag_system:
        st.error("System unavailable. Please refresh the page.")
        return
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    
    # Enhanced interface
    EnhancedUIComponents.render_enhanced_header()
    
    # Get system analytics
    analytics = rag_system.get_system_analytics()
    
    # Enhanced sidebar
    EnhancedUIComponents.render_enhanced_sidebar(rag_system.system_mode, analytics)
    
    # Main query interface
    st.markdown("### 💬 Ask Your Swimming Question")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_query = st.text_area(
            "",
            height=80,
            placeholder="Example: How can I improve my freestyle breathing technique for competitive swimming?",
            help="Ask detailed swimming questions for best results. The system includes advanced evaluation and feedback."
        )
        
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            ask_button = st.button("🏊‍♂️ Get Expert Advice", type="primary")
        with col_btn2:
            clear_button = st.button("🗑️ Clear History")
        with col_btn3:
            example_button = st.button("💡 Show Example")
        with col_btn4:
            analytics_button = st.button("📊 View Analytics")
    
    with col2:
        st.markdown("### 💡 Enhanced Features")
        st.info("""
        **New in this version:**
        - Reference-based evaluation
        - Hallucination detection
        - Advanced reranking
        - User feedback system
        - Degraded mode handling
        """)
        
        # Quick suggestions
        suggestions = [
            "Improve freestyle endurance",
            "Perfect butterfly timing", 
            "Master breaststroke coordination",
            "Backstroke technique refinement",
            "Competitive start improvements"
        ]
        
        suggestion_clicked = ""
        st.markdown("**Quick Suggestions:**")
        for i, suggestion in enumerate(suggestions):
            if st.button(f"🎯 {suggestion}", key=f"sugg_{i}"):
                suggestion_clicked = suggestion
    
    # Handle actions
    final_query = user_query
    if example_button or suggestion_clicked:
        if example_button:
            final_query = "How can I develop better bilateral breathing in freestyle while maintaining speed and efficiency for competitive swimming events?"
        else:
            final_query = suggestion_clicked
        ask_button = True
    
    # Clear history
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
        st.success("🗑️ Chat history cleared!")
        st.rerun()
    
    # Show analytics
    if analytics_button:
        st.markdown("### 📊 System Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("System Mode", analytics['system_mode'].upper())
            st.metric("Components Ready", sum(analytics.get('component_status', {}).values()))
        
        with col2:
            if analytics.get('feedback'):
                feedback_data = analytics['feedback']
                st.metric("Total Feedback", feedback_data.get('total_feedback', 0))
                st.metric("Avg Rating", f"{feedback_data.get('avg_rating', 0):.1f}/5")
        
        with col3:
            st.metric("Active Session", st.session_state.session_id)
            st.metric("Queries This Session", len(st.session_state.chat_history))
        
        # Component status
        if analytics.get('component_status'):
            st.markdown("**Component Status:**")
            components = analytics['component_status']
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                component_name = component.replace('_', ' ').title()
                st.write(f"{status_icon} {component_name}")
    
    # Process query
    if (ask_button or suggestion_clicked) and final_query.strip():
        with st.spinner("🔄 Processing your enhanced swimming question..."):
            try:
                # Process with enhanced system
                response = rag_system.process_query(
                    final_query, 
                    session_id=st.session_state.session_id,
                    user_context={'interface': 'streamlit'}
                )
                
                # Add to history
                chat_entry = {
                    'question': final_query,
                    'response': response,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'settings': {
                        'show_evaluation': st.session_state.get('show_evaluation', True),
                        'show_details': st.session_state.get('show_details', False),
                        'show_images': st.session_state.get('show_images', True),
                        'enable_feedback': st.session_state.get('enable_feedback', True)
                    }
                }
                
                st.session_state.chat_history.append(chat_entry)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                logger.error(f"Enhanced processing error: {e}")
    
    # Enhanced chat history display
    if st.session_state.chat_history:
        st.markdown("### 🏊‍♂️ Your Enhanced Coaching Session")
        
        for chat in reversed(st.session_state.chat_history[-10:]):  # Show last 10
            EnhancedUIComponents.render_chat_entry_enhanced(
                chat, rag_system.image_processor, rag_system
            )
            st.divider()
    else:
        st.markdown("### 👋 Welcome to Enhanced Swimming Coach AI")
        st.info("""
        This advanced RAG system includes:
        
        **🔍 Enhanced Evaluation:**
        - Reference dataset comparison
        - Hallucination detection
        - Multi-dimensional scoring
        
        **⚡ Improved Performance:**
        - Advanced reranking
        - Intelligent query expansion
        - Hybrid search methods
        
        **📊 User Experience:**
        - Feedback collection
        - Performance analytics
        - Degraded mode handling
        
        Ask your first swimming question to see these features in action!
        """)
    
    # Enhanced footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%); 
                color: white; text-align: center; padding: 2rem; 
                border-radius: 12px; margin-top: 2rem;">
        <h4>🏊‍♂️ Enhanced Swimming Coach AI - Production RAG System</h4>
        <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1rem; flex-wrap: wrap;">
            <div>
                <strong>✨ Reference Evaluation</strong><br>
                <small>Benchmark against expert knowledge</small>
            </div>
            <div>
                <strong>🛡️ Hallucination Detection</strong><br>
                <small>Advanced fact-checking system</small>
            </div>
            <div>
                <strong>📊 User Feedback</strong><br>
                <small>Continuous improvement loop</small>
            </div>
            <div>
                <strong>🎯 Smart Reranking</strong><br>
                <small>Multi-signal result optimization</small>
            </div>
        </div>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.9;">
            🏗️ Modular Architecture • 📈 Comprehensive Analytics • 🔄 Degraded Mode Support • 🚀 Production Ready
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Periodic cleanup
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-30:]
    
    # Memory management
    if hasattr(st.session_state, 'cleanup_counter'):
        st.session_state.cleanup_counter += 1
    else:
        st.session_state.cleanup_counter = 1
    
    if st.session_state.cleanup_counter % 15 == 0:
        gc.collect()

# ==================== HELPER CLASSES ====================

class LLMProvider:
    """Enhanced LLM provider with better prompting"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        self.system_prompt = """You are an expert swimming coach AI with comprehensive knowledge of swimming techniques, training methods, and performance optimization.

Your responses should be:
- **Practical and Actionable**: Provide specific steps and drills
- **Technically Accurate**: Use proper swimming terminology
- **Well-Structured**: Organize information clearly
- **Safety-Conscious**: Always prioritize swimmer safety
- **Evidence-Based**: Reference proper techniques and methods

Response format:
- 250-400 words for comprehensive coverage
- Clear headings or numbered steps when appropriate
- Include specific drills or exercises when relevant
- Maintain encouraging and professional tone
- End with progression advice when applicable

Focus on the provided documentation and ensure all advice is practical and implementable."""
    
    @lru_cache(maxsize=1)
    def get_api_key(self) -> Optional[str]:
        """Get API key with caching"""
        try:
            api_key = st.secrets["GROK"]["api_key"].strip()
            return api_key
        except KeyError:
            api_key = os.environ.get('GROK_API_KEY')
            if api_key:
                return api_key.strip()
            else:
                logger.error("Grok API key not found")
                return None
    
    def generate_response(self, prompt: str) -> str:
        """Generate response with enhanced error handling"""
        api_key = self.get_api_key()
        if not api_key:
            return "I apologize, but the system is not properly configured. Please contact support."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.config.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
                if response.status_code == 401:
                    logger.error("Grok API key rejected (401 Unauthorized). Check your key!")
                    return "Authentication failed. Please check your API key."
            
                logger.warning(f"API error {response.status_code}, attempt {attempt + 1}")
                if attempt == self.config.max_retries - 1:
                    return f"I'm experiencing technical difficulties (API Error {response.status_code}). Please try again."
            
            except requests.exceptions.Timeout:
                if attempt == self.config.max_retries - 1:
                    return "The system is experiencing high load. Please try a shorter question."
                
            except Exception as e:
                logger.error(f"Generation error attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    return "Technical error occurred. Please try again."
        
            # Exponential backoff
            time.sleep(2 ** attempt)
    
        return "Unable to generate response. Please try again."
    
    def build_prompt(self, query: str, search_results: List[Dict]) -> str:
        """Build enhanced prompt with better context"""
        context_parts = []
        
        if search_results:
            doc_context = []
            
            for i, result in enumerate(search_results[:4], 1):  # Top 4 results
                source = result.get('pdf', 'Swimming Guide').replace('.pdf', '').replace('_', ' ')
                stroke = result.get('stroke', 'General')
                swim_type = result.get('swimming_type', 'General').replace('Swimming-', '')
                similarity = result.get('similarity_score', 0)
                text = result['text'][:500]  # Limit text length
                
                doc_header = f"SOURCE {i}: {source} - {stroke} {swim_type} (relevance: {similarity:.2f})"
                doc_context.append(f"{doc_header}\n{text}")
            
            context_parts.append("SWIMMING DOCUMENTATION:\n" + '\n\n'.join(doc_context))
            
            # Visual aids information
            total_images = sum(len(result.get('images', [])) for result in search_results)
            if total_images > 0:
                context_parts.append(f"VISUAL AIDS: {total_images} demonstration images available for reference")
        
        full_context = '\n\n'.join(context_parts) if context_parts else "Provide general swimming guidance based on expert knowledge."
        
        return f"""{full_context}

SWIMMER'S QUESTION: "{query}"

Please provide expert swimming coaching advice based on the documentation above. Include specific techniques, actionable steps, and practical drills where appropriate. Ensure your response is comprehensive yet focused on the swimmer's specific question."""

# ==================== SYSTEM LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swimming_rag.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    main()
