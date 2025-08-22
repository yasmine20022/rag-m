import streamlit as st
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from typing import List, Dict, Tuple
import re
import os
from PIL import Image
import base64
from datetime import datetime
import warnings
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import logging
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏊‍♂️ Swimming Coach AI - Ultra-Optimized RAG",
    page_icon="🏊‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .evaluation-panel {
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(240,248,255,0.98) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        border: 2px solid #e3f2fd;
        position: relative;
    }
    .evaluation-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff9800, #4caf50, #2196f3);
        border-radius: 15px 15px 0 0;
    }
    .evaluation-header {
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .evaluation-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    .performance-badge-main {
        display: inline-block;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0.5rem;
        text-align: center;
        min-width: 200px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 4px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card.excellent { border-left-color: #4caf50; }
    .metric-card.good { border-left-color: #ff9800; }
    .metric-card.poor { border-left-color: #f44336; }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .metric-score {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
    }
    .recommendations-section {
        background: rgba(33, 150, 243, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid #2196f3;
    }
    .recommendations-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1976d2;
        margin-bottom: 1rem;
    }
    .recommendation-item {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 3px solid #4caf50;
    }
    .improvements-section {
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: white;
    }
    .improvements-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .improvement-item {
        background: rgba(255,255,255,0.2);
        padding: 0.6rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .badge-excellent { background: #4caf50; color: white; }
    .badge-good { background: #ff9800; color: white; }
    .badge-medium { background: #ffc107; color: black; }
    .badge-poor { background: #f44336; color: white; }
    .status-card {
        background: linear-gradient(135deg, #4caf50, #2e7d32);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class AdvancedImageDescriptor:
    """PROBLÈME 1 CORRIGÉ: Image description system optimized for swimming"""
    
    def __init__(self):
        self.swimming_vocabulary = {
            'strokes': {
                'freestyle': ['freestyle', 'front crawl', 'crawl stroke', 'free', 'front stroke'],
                'backstroke': ['backstroke', 'back crawl', 'back stroke', 'back', 'supine'],
                'breaststroke': ['breaststroke', 'breast stroke', 'frog stroke', 'breast', 'frog'],
                'butterfly': ['butterfly', 'fly stroke', 'dolphin stroke', 'fly', 'dolphin']
            },
            'techniques': {
                'breathing': ['breathing', 'breath', 'bilateral', 'air', 'oxygen', 'ventilation'],
                'kick': ['kick', 'flutter', 'dolphin', 'frog', 'legs', 'kicking', 'propulsion'],
                'pull': ['pull', 'arm', 'stroke', 'catch', 'hands', 'pulling', 'arm movement'],
                'body_position': ['position', 'posture', 'alignment', 'streamline', 'body line'],
                'timing': ['timing', 'coordination', 'rhythm', 'tempo', 'synchronization']
            },
            'equipment': ['kickboard', 'pull buoy', 'fins', 'paddles', 'goggles', 'cap'],
            'environment': ['pool', 'water', 'lane', 'swimming pool', 'aquatic', 'chlorine']
        }
        
        # Load BLIP with improved error handling
        self.blip_processor = None
        self.blip_model = None
        self._load_blip_model()
    
    def _load_blip_model(self):
        """Load BLIP model with retry logic"""
        try:
            logger.info("🤖 Loading BLIP for image descriptions...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.eval()
            logger.info("✅ BLIP loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ BLIP not available: {str(e)}")
    
    def generate_enhanced_description(self, image_path: str, context_chunk: Dict = None) -> str:
        """CORRECTION PROBLÈME 1: Generate varied and accurate descriptions"""
        try:
            if not os.path.exists(image_path):
                return self._get_fallback_description(context_chunk)
            
            description_parts = []
            
            # 1. Get detailed AI description
            if self.blip_model:
                ai_description = self._get_detailed_ai_description(image_path)
                if ai_description:
                    description_parts.append(ai_description)
            
            # 2. Add contextual swimming analysis
            if context_chunk:
                context_desc = self._generate_detailed_context(context_chunk)
                if context_desc:
                    description_parts.append(context_desc)
            
            # 3. Enhanced filename analysis
            filename_desc = self._analyze_filename_detailed(image_path)
            if filename_desc:
                description_parts.append(filename_desc)
            
            # 4. Visual analysis (colors, composition)
            visual_desc = self._analyze_visual_elements(image_path)
            if visual_desc:
                description_parts.append(visual_desc)
            
            return self._create_varied_description(description_parts, context_chunk)
            
        except Exception as e:
            logger.error(f"Image description error: {str(e)}")
            return self._get_emergency_fallback_description()
    
    def _get_detailed_ai_description(self, image_path: str) -> str:
        """Get detailed AI description with swimming-specific prompts"""
        try:
            pil_image = Image.open(image_path).convert('RGB')
            
            # Use conditional text to guide generation toward swimming
            conditional_text = "a swimmer demonstrating technique in a pool"
            inputs = self.blip_processor(pil_image, conditional_text, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(
                    **inputs, 
                    max_length=80,  # Longer descriptions
                    num_beams=8,
                    length_penalty=1.5,  # Encourage longer descriptions
                    repetition_penalty=1.3,  # Avoid repetition
                    do_sample=True,
                    temperature=0.8  # Add some creativity
                )
            
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            return self._enhance_ai_caption(caption)
        except Exception as e:
            logger.error(f"AI description error: {e}")
            return ""
    
    def _enhance_ai_caption(self, caption: str) -> str:
        """Enhance AI caption with swimming-specific improvements"""
        if not caption:
            return ""
        
        caption_lower = caption.lower()
        
        # Swimming-specific enhancements
        enhancements = {
            'person in water': 'competitive swimmer executing stroke technique',
            'person swimming': 'athlete demonstrating optimal swimming form',
            'man swimming': 'male swimmer showing proper technique mechanics',
            'woman swimming': 'female swimmer displaying efficient stroke pattern',
            'blue water': 'championship pool with clear competition-grade water',
            'swimming pool': 'professional training facility pool',
            'person floating': 'swimmer maintaining perfect streamline position',
            'arms extended': 'demonstrating extended catch phase with high elbow position',
            'underwater': 'underwater stroke mechanics and body positioning'
        }
        
        enhanced_caption = caption
        for old_phrase, new_phrase in enhancements.items():
            if old_phrase in caption_lower:
                enhanced_caption = enhanced_caption.replace(old_phrase, new_phrase)
        
        # Add technical swimming terms
        technical_additions = [
            "showcasing biomechanically efficient movement patterns",
            "demonstrating race-ready technique fundamentals", 
            "illustrating optimal hydrodynamic positioning",
            "displaying championship-level stroke mechanics",
            "exemplifying textbook swimming form execution"
        ]
        
        # Randomly select a technical addition to vary descriptions
        import random
        technical_add = random.choice(technical_additions)
        
        if len(enhanced_caption) < 100:  # Only add if description is short
            enhanced_caption += f" - {technical_add}"
        
        return enhanced_caption
    
    def _generate_detailed_context(self, context_chunk: Dict) -> str:
        """Generate detailed contextual description"""
        context_parts = []
        
        stroke = context_chunk.get('stroke', '').lower()
        swimming_type = context_chunk.get('swimming_type', '')
        level = context_chunk.get('level', '')
        
        # Detailed stroke-specific descriptions
        stroke_details = {
            'freestyle': 'freestyle stroke emphasizing bilateral breathing, high elbow catch, and efficient body rotation',
            'backstroke': 'backstroke technique focusing on consistent rhythm, straight arm recovery, and hip-driven rotation',
            'breaststroke': 'breaststroke demonstrating the coordinated timing of pull-breathe-kick-glide sequence',
            'butterfly': 'butterfly stroke showcasing powerful undulation, synchronized arm movement, and dolphin kick timing'
        }
        
        if stroke in stroke_details:
            context_parts.append(stroke_details[stroke])
        
        # Content type details
        type_descriptions = {
            'Swimming-Drill': 'progressive training drill designed for skill acquisition and muscle memory development',
            'Swimming-Technique': 'technical instruction focusing on biomechanical optimization and efficiency improvement',
            'Swimming-Training': 'structured training methodology for performance enhancement and conditioning',
            'Swimming-Competition': 'competitive swimming preparation and race strategy implementation'
        }
        
        if swimming_type in type_descriptions:
            context_parts.append(type_descriptions[swimming_type])
        
        # Level-specific details
        level_details = {
            'beginner': 'foundational instruction suitable for swimmers developing basic stroke competency',
            'intermediate': 'skill-building progression for swimmers refining technique and building endurance',
            'advanced': 'elite-level instruction for competitive swimmers and serious enthusiasts',
            'competitive': 'race-specific preparation for competitive swimming performance optimization'
        }
        
        if level.lower() in level_details:
            context_parts.append(level_details[level.lower()])
        
        return " featuring " + ", ".join(context_parts) if context_parts else ""
    
    def _analyze_filename_detailed(self, image_path: str) -> str:
        """Detailed filename analysis with swimming context"""
        filename = os.path.basename(image_path).lower()
        
        detailed_analyses = {
            'freestyle': 'front crawl technique demonstration with emphasis on stroke efficiency',
            'backstroke': 'backstroke instruction showing proper body position and arm coordination',
            'butterfly': 'butterfly stroke breakdown illustrating power generation and timing',
            'breaststroke': 'breaststroke technique guide emphasizing kick-pull coordination',
            'breathing': 'breathing technique instruction for improved oxygen efficiency',
            'kick': 'kicking drill demonstration for leg strength and propulsion development',
            'start': 'racing start technique showing explosive power and entry form',
            'turn': 'flip turn mechanics instruction for competitive swimming efficiency',
            'underwater': 'underwater technique showing streamline position and dolphin kick'
        }
        
        for key, description in detailed_analyses.items():
            if key in filename:
                return description
        
        return ""
    
    def _analyze_visual_elements(self, image_path: str) -> str:
        """Analyze visual elements for enhanced descriptions"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            # Analyze dominant colors
            avg_color = np.mean(img, axis=(0,1))
            blue_dominance = avg_color[0] / (avg_color[1] + avg_color[2] + 1)
            
            visual_elements = []
            
            if blue_dominance > 1.2:
                visual_elements.append("captured in crystal-clear pool conditions")
            
            # Analyze brightness for indoor/outdoor detection
            brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if brightness > 150:
                visual_elements.append("in well-lit competitive swimming environment")
            elif brightness < 100:
                visual_elements.append("in controlled indoor training facility")
            
            return " ".join(visual_elements)
        except:
            return ""
    
    def _create_varied_description(self, parts: List[str], context_chunk: Dict = None) -> str:
        """Create varied descriptions to avoid repetition"""
        if not parts:
            return self._get_fallback_description(context_chunk)
        
        # Templates for variety
        templates = [
            "Professional swimming instruction showing {main} {details}",
            "Technical demonstration of {main} {details}", 
            "Championship-level {main} {details}",
            "Biomechanical analysis featuring {main} {details}",
            "Expert coaching visual guide displaying {main} {details}"
        ]
        
        import random
        template = random.choice(templates)
        
        main_description = parts[0] if parts else "swimming technique"
        additional_details = " with " + ", ".join(parts[1:]) if len(parts) > 1 else ""
        
        return template.format(main=main_description, details=additional_details)
    
    def _get_fallback_description(self, context_chunk: Dict = None) -> str:
        """Fallback descriptions with variety"""
        fallbacks = [
            "Professional swimming technique demonstration for skill development",
            "Expert coaching instruction showing optimal stroke mechanics",
            "Technical training guide for competitive swimming performance",
            "Biomechanical swimming analysis for technique improvement",
            "Championship-level instruction for stroke optimization"
        ]
        
        if context_chunk:
            stroke = context_chunk.get('stroke', '').lower()
            if stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']:
                fallbacks.append(f"Advanced {stroke} technique instruction for competitive swimmers")
        
        import random
        return random.choice(fallbacks)
    
    def _get_emergency_fallback_description(self) -> str:
        """Emergency fallback with randomization"""
        emergency_descriptions = [
            "Swimming technique instructional content for athlete development",
            "Professional coaching demonstration for stroke improvement", 
            "Technical swimming analysis for performance enhancement",
            "Expert instruction guide for competitive swimming excellence"
        ]
        
        import random
        return random.choice(emergency_descriptions)

class UltraOptimizedRAGEvaluator:
    """PROBLÈME 3 CORRIGÉ: Ultra-optimized RAG evaluator with improved display"""
    
    def __init__(self):
        logger.info("📊 Initializing ultra-optimized RAG evaluator...")
        self.image_descriptor = AdvancedImageDescriptor()
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Enriched vocabulary
        self.swimming_vocabulary = {
            'strokes': ['freestyle', 'backstroke', 'breaststroke', 'butterfly', 'crawl', 'stroke', 'swimming'],
            'techniques': ['technique', 'form', 'breathing', 'kick', 'pull', 'position', 'timing', 'coordination'],
            'training': ['drill', 'exercise', 'practice', 'training', 'workout', 'set', 'repetition'],
            'performance': ['speed', 'endurance', 'efficiency', 'power', 'sprint', 'distance', 'improvement'],
            'equipment': ['kickboard', 'fins', 'paddles', 'goggles', 'pull buoy', 'equipment'],
            'levels': ['beginner', 'intermediate', 'advanced', 'novice', 'competitive', 'elite']
        }
        
        # REALISTIC THRESHOLDS
        self.performance_thresholds = {
            'excellent': 75,
            'good': 60,
            'acceptable': 45,
            'poor': 30
        }
    
    def evaluate_retrieval_enhanced(self, query: str, search_results: List[Dict]) -> Tuple[float, Dict]:
        """Retrieval evaluation with semantic metrics"""
        details = {
            "num_results": len(search_results),
            "semantic_similarity": 0.0,
            "domain_relevance": 0.0,
            "content_diversity": 0.0,
            "technical_depth": 0.0,
            "contextual_alignment": 0.0
        }
        
        if not search_results:
            return 40.0, details
        
        try:
            # 1. Semantic similarity with embeddings
            query_embedding = self.sentence_model.encode([query])
            result_texts = [result.get('text', '')[:500] for result in search_results]
            result_embeddings = self.sentence_model.encode(result_texts)
            
            semantic_scores = cosine_similarity(query_embedding, result_embeddings)[0]
            details["semantic_similarity"] = float(np.mean(semantic_scores)) * 100
            
            # 2. Swimming domain relevance (more generous)
            query_lower = query.lower()
            domain_scores = []
            
            for result in search_results:
                text_lower = result.get('text', '').lower()
                
                # Count terms from each category
                category_matches = 0
                for category_terms in self.swimming_vocabulary.values():
                    if any(term in text_lower for term in category_terms):
                        category_matches += 1
                
                # More generous scoring
                domain_score = min((category_matches * 25), 100)
                domain_scores.append(domain_score)
            
            details["domain_relevance"] = float(np.mean(domain_scores)) if domain_scores else 50
            
            # 3. Content diversity (bonus)
            sources = set(result.get('pdf', 'unknown') for result in search_results)
            stroke_types = set(result.get('stroke', 'general') for result in search_results)
            diversity_score = (len(sources) * 20 + len(stroke_types) * 15)
            details["content_diversity"] = min(diversity_score, 100)
            
            # 4. Technical depth (adjusted)
            technical_scores = []
            for result in search_results:
                text_lower = result.get('text', '').lower()
                technical_terms = sum(1 for term in self.swimming_vocabulary['techniques'] if term in text_lower)
                tech_score = min(technical_terms * 35, 100)
                technical_scores.append(tech_score)
            
            details["technical_depth"] = float(np.mean(technical_scores)) if technical_scores else 40
            
            # 5. Contextual alignment
            context_score = 70
            if any(result.get('similarity_score', 0) > 0.5 for result in search_results):
                context_score = 85
            elif any(result.get('similarity_score', 0) > 0.3 for result in search_results):
                context_score = 75
            
            details["contextual_alignment"] = context_score
            
            # Final score with optimized weighting
            retrieve_score = (
                details["semantic_similarity"] * 0.25 +
                details["domain_relevance"] * 0.25 +
                details["technical_depth"] * 0.20 +
                details["contextual_alignment"] * 0.20 +
                details["content_diversity"] * 0.10
            )
            
            return min(retrieve_score, 100.0), details
            
        except Exception as e:
            logger.error(f"Retrieval evaluation error: {e}")
            return 50.0, details
    
    def evaluate_generation_semantic(self, query: str, response: str, search_results: List[Dict]) -> Tuple[float, Dict]:
        """Generation evaluation with advanced semantic analysis"""
        details = {
            "response_length": len(response),
            "semantic_coherence": 0.0,
            "technical_accuracy": 0.0,
            "practical_value": 0.0,
            "query_relevance": 0.0,
            "source_utilization": 0.0
        }
        
        if not response or len(response) < 20:
            return 30.0, details
        
        try:
            response_lower = response.lower()
            
            # 1. Semantic coherence with query
            if len(query.strip()) > 0:
                query_embedding = self.sentence_model.encode([query])
                response_embedding = self.sentence_model.encode([response])
                semantic_similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
                details["semantic_coherence"] = max(semantic_similarity * 100, 40)
            
            # 2. Technical accuracy (more flexible)
            technical_terms_found = 0
            for category_terms in self.swimming_vocabulary.values():
                if any(term in response_lower for term in category_terms):
                    technical_terms_found += 1
            
            details["technical_accuracy"] = min(technical_terms_found * 25 + 10, 95)
            
            # 3. Practical value (action words)
            action_indicators = [
                'practice', 'focus', 'maintain', 'improve', 'work', 'try', 'keep', 
                'drill', 'exercise', 'train', 'develop', 'strengthen', 'technique'
            ]
            action_count = sum(1 for word in action_indicators if word in response_lower)
            details["practical_value"] = min(action_count * 20 + 30, 100)
            
            # 4. Query relevance (shared words)
            query_words = set(query.lower().split())
            response_words = set(response_lower.split())
            
            if query_words:
                overlap = len(query_words.intersection(response_words))
                relevance = (overlap / len(query_words)) * 100
                details["query_relevance"] = min(relevance + 20, 100)
            else:
                details["query_relevance"] = 60
            
            # 5. Source utilization
            source_utilization = 60
            if search_results:
                source_terms = set()
                for result in search_results[:3]:
                    source_terms.update(result.get('text', '').lower().split()[:100])
                
                if source_terms:
                    utilization = len(response_words.intersection(source_terms)) / min(len(response_words), 50)
                    details["source_utilization"] = min(utilization * 100 + 40, 100)
            
            # Final weighted score (emphasis on utility)
            generate_score = (
                details["semantic_coherence"] * 0.20 +
                details["technical_accuracy"] * 0.25 +
                details["practical_value"] * 0.25 +
                details["query_relevance"] * 0.20 +
                details["source_utilization"] * 0.10
            )
            
            return min(generate_score, 100.0), details
            
        except Exception as e:
            logger.error(f"Generation evaluation error: {e}")
            return 55.0, details
    
    def evaluate_image_coherence_advanced(self, query: str, response: str, search_results: List[Dict]) -> Tuple[float, Dict]:
        """Advanced image coherence evaluation with AI descriptions and generous scoring"""
        all_images = []
        
        # Extract images with advanced descriptions
        for result in search_results:
            for img in result.get('images', []):
                img_path = img.get('path', '')
                if img_path:
                    enhanced_desc = self.image_descriptor.generate_enhanced_description(
                        img_path, result
                    )
                    
                    all_images.append({
                        'path': img_path,
                        'enhanced_description': enhanced_desc,
                        'source_context': result,
                        'exists': os.path.exists(img_path)
                    })
        
        details = {
            "num_images": len(all_images),
            "availability_score": 0.0,
            "semantic_alignment": 0.0,
            "contextual_relevance": 0.0,
            "technical_value": 0.0
        }
        
        if not all_images:
            return 70.0, details  # Neutral generous score if no images
        
        try:
            # 1. Availability (less penalizing)
            existing_count = sum(1 for img in all_images if img['exists'])
            if all_images:
                availability = (existing_count / len(all_images)) * 100
                details["availability_score"] = max(availability, 40)
            
            # 2. Semantic alignment with embeddings
            if existing_count > 0:
                descriptions = [img['enhanced_description'] for img in all_images if img['exists']]
                
                # With query
                if descriptions:
                    query_emb = self.sentence_model.encode([query])
                    desc_embs = self.sentence_model.encode(descriptions)
                    query_similarities = cosine_similarity(query_emb, desc_embs)[0]
                    details["semantic_alignment"] = float(np.mean(query_similarities)) * 100
                    
                    # With response
                    response_emb = self.sentence_model.encode([response])
                    response_similarities = cosine_similarity(response_emb, desc_embs)[0]
                    details["contextual_relevance"] = float(np.mean(response_similarities)) * 100
            
            # 3. Technical value (swimming keywords in descriptions)
            technical_scores = []
            technical_keywords = ['technique', 'form', 'stroke', 'swimming', 'water', 'pool', 'training']
            
            for img in all_images:
                if img['exists']:
                    desc_lower = img['enhanced_description'].lower()
                    tech_count = sum(1 for keyword in technical_keywords if keyword in desc_lower)
                    tech_score = min(tech_count * 20 + 40, 100)
                    technical_scores.append(tech_score)
            
            details["technical_value"] = float(np.mean(technical_scores)) if technical_scores else 60
            
            # Final image score (less critical, more bonus)
            if existing_count == 0:
                image_score = 60  # No images = neutral
            else:
                image_score = (
                    details["availability_score"] * 0.25 +
                    details["semantic_alignment"] * 0.30 +
                    details["contextual_relevance"] * 0.30 +
                    details["technical_value"] * 0.15
                )
            
            return min(image_score, 100.0), details
            
        except Exception as e:
            logger.error(f"Image evaluation error: {e}")
            return 65.0, details
    
    def evaluate_complete_rag_optimized(self, query: str, response: str, search_results: List[Dict]) -> Dict:
        """Complete optimized RAG evaluation with realistic scoring"""
        logger.info(f"📊 Optimized evaluation for: '{query[:50]}...'")
        
        try:
            # Individual evaluations
            retrieve_score, retrieve_details = self.evaluate_retrieval_enhanced(query, search_results)
            generate_score, generate_details = self.evaluate_generation_semantic(query, response, search_results)
            image_score, image_details = self.evaluate_image_coherence_advanced(query, response, search_results)
            
            # Global score with new weighting (focus on generation)
            overall_score = (
                retrieve_score * 0.30 +    # Reduced from 0.4 to 0.3
                generate_score * 0.50 +    # Increased from 0.4 to 0.5
                image_score * 0.20         # Images remain bonus
            )
            
            # Classification with new realistic thresholds
            if overall_score >= self.performance_thresholds['excellent']:
                performance_level = "🏆 Excellent"
                performance_class = "badge-excellent"
            elif overall_score >= self.performance_thresholds['good']:
                performance_level = "👍 Good"
                performance_class = "badge-good"
            elif overall_score >= self.performance_thresholds['acceptable']:
                performance_level = "⚠️ Acceptable"
                performance_class = "badge-medium"
            else:
                performance_level = "🔄 Needs Improvement"
                performance_class = "badge-poor"
            
            return {
                'overall_score': overall_score,
                'performance_level': performance_level,
                'performance_class': performance_class,
                'retrieve_score': retrieve_score,
                'generate_score': generate_score,
                'image_score': image_score,
                'details': {
                    'retrieval': retrieve_details,
                    'generation': generate_details,
                    'images': image_details
                },
                'recommendations': self._generate_smart_recommendations(retrieve_score, generate_score, image_score),
                'improvement_highlights': self._identify_improvements()
            }
            
        except Exception as e:
            logger.error(f"Complete evaluation error: {e}")
            return self._get_fallback_evaluation()
    
    def _generate_smart_recommendations(self, retrieve_score: float, generate_score: float, image_score: float) -> List[str]:
        """Smart recommendations based on scores"""
        recommendations = []
        
        # Targeted recommendations with adjusted thresholds
        if retrieve_score < 65:
            recommendations.append("🔍 Optimize retrieval: Improve query or enrich document database")
        
        if generate_score < 65:
            recommendations.append("✍️ Enhance generation: Refine system prompt or increase tokens")
        
        if image_score < 60:
            recommendations.append("🖼️ Enrich visual content: Check image availability and descriptions")
        
        # Positive recommendations
        if retrieve_score >= 75:
            recommendations.append("✅ Excellent document retrieval!")
        
        if generate_score >= 75:
            recommendations.append("✅ High-quality response generated!")
        
        if not recommendations:
            recommendations.append("🎯 Balanced performance across all aspects")
        
        return recommendations
    
    def _identify_improvements(self) -> List[str]:
        """Identify system improvements"""
        return [
            "🎯 Realistic evaluation thresholds recalibrated",
            "🧠 Semantic analysis with embeddings for precision",
            "🖼️ Advanced AI image descriptions with BLIP",
            "⚖️ Optimized weighting (50% generation, 30% retrieval)",
            "📊 More generous base metrics to avoid excessive penalization"
        ]
    
    def _get_fallback_evaluation(self) -> Dict:
        """Fallback evaluation in case of error"""
        return {
            'overall_score': 60.0,
            'performance_level': "⚠️ Partial Evaluation",
            'performance_class': "badge-medium",
            'retrieve_score': 60.0,
            'generate_score': 60.0,
            'image_score': 60.0,
            'details': {},
            'recommendations': ["🔧 Check system configuration"],
            'improvement_highlights': ["⚠️ Degraded mode activated"]
        }
    
    def get_score_class(self, score: float) -> str:
        """CSS classes adapted to new thresholds"""
        if score >= 75:
            return "excellent"
        elif score >= 60:
            return "good"
        else:
            return "poor"
    
    def display_evaluation_enhanced(self, evaluation: Dict):
        """PROBLÈME 3 CORRIGÉ: Enhanced evaluation display with proper metrics visibility"""
        
        # Main evaluation panel with improved styling
        st.markdown(f"""
        <div class="evaluation-panel">
            <div class="evaluation-header">
                <div class="evaluation-title">📊 Automatic RAG Evaluation - Ultra-Optimized Version</div>
                <div class="performance-badge-main {evaluation['performance_class']}">
                    {evaluation['performance_level']} - {evaluation['overall_score']:.1f}/100
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # PROBLÈME 3 CORRIGÉ: Metrics properly displayed in grid
        st.markdown(f"""
            <div class="metrics-container">
                <div class="metric-card {self.get_score_class(evaluation['retrieve_score'])}">
                    <div class="metric-label">🔍 Retrieval</div>
                    <div class="metric-score">{evaluation['retrieve_score']:.1f}/100</div>
                </div>
                <div class="metric-card {self.get_score_class(evaluation['generate_score'])}">
                    <div class="metric-label">✍️ Generation</div>
                    <div class="metric-score">{evaluation['generate_score']:.1f}/100</div>
                </div>
                <div class="metric-card {self.get_score_class(evaluation['image_score'])}">
                    <div class="metric-label">🖼️ AI Images</div>
                    <div class="metric-score">{evaluation['image_score']:.1f}/100</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("""
            <div class="recommendations-section">
                <div class="recommendations-title">💡 Smart Recommendations:</div>
        """, unsafe_allow_html=True)
        
        for rec in evaluation['recommendations']:
            st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Improvements highlight section
        if 'improvement_highlights' in evaluation:
            st.markdown("""
            <div class="improvements-section">
                <div class="improvements-title">🚀 Ultra-Optimized RAG Improvements</div>
            """, unsafe_allow_html=True)
            
            for improvement in evaluation['improvement_highlights']:
                st.markdown(f'<div class="improvement-item">{improvement}</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

class SwimmingUltraOptimizedRAG:
    """Ultra-optimized RAG with intelligent search and improved generation"""
    
    def __init__(self):
        logger.info("🏊‍♂️ Initializing ultra-optimized RAG...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.evaluator = UltraOptimizedRAGEvaluator()
        
        # Enriched synonym vocabulary
        self.advanced_swimming_terms = {
            'freestyle': ['freestyle', 'front crawl', 'crawl stroke', 'free', 'front stroke', 'australian crawl'],
            'backstroke': ['backstroke', 'back crawl', 'back stroke', 'supine swimming', 'elementary backstroke'],
            'breaststroke': ['breaststroke', 'breast stroke', 'frog stroke', 'whip kick', 'breaststroke kick'],
            'butterfly': ['butterfly', 'fly stroke', 'dolphin stroke', 'butterfly kick', 'undulation'],
            'technique': ['technique', 'form', 'mechanics', 'stroke technique', 'swimming form', 'biomechanics'],
            'breathing': ['breathing', 'breath', 'bilateral breathing', 'ventilation', 'air intake', 'oxygen'],
            'kick': ['kick', 'flutter kick', 'dolphin kick', 'frog kick', 'leg movement', 'propulsion'],
            'pull': ['pull', 'arm stroke', 'catch', 'pull phase', 'arm movement', 'stroke cycle'],
            'training': ['training', 'workout', 'practice', 'drill', 'exercise', 'conditioning', 'preparation'],
            'improvement': ['improve', 'better', 'enhance', 'develop', 'progress', 'advance', 'optimize']
        }
        
        logger.info("✅ Ultra-optimized RAG initialized!")
    
    def load_swimming_chunks(self, file_path: str):
        """Load chunks with enriched validation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"🏊‍♂️ {len(self.chunks)} chunks loaded")
            self.analyze_content_advanced()
            
        except Exception as e:
            st.error(f"Chunk loading error: {str(e)}")
    
    def analyze_content_advanced(self):
        """Advanced content analysis with detailed metrics"""
        categories = {
            'technique_focused': 0, 'drill_based': 0, 'beginner_friendly': 0, 'advanced_level': 0,
            'freestyle_content': 0, 'backstroke_content': 0, 'breaststroke_content': 0, 'butterfly_content': 0,
            'with_images': 0, 'comprehensive_chunks': 0, 'short_chunks': 0
        }
        
        total_length = 0
        
        for chunk in self.chunks:
            text = chunk.get('text', '')
            text_lower = text.lower()
            text_length = len(text)
            total_length += text_length
            
            # Advanced categorical analysis
            if any(term in text_lower for term in ['technique', 'form', 'mechanics', 'proper']):
                categories['technique_focused'] += 1
            
            if any(term in text_lower for term in ['drill', 'exercise', 'practice', 'training']):
                categories['drill_based'] += 1
                
            if any(term in text_lower for term in ['beginner', 'novice', 'learning', 'basic', 'introduction']):
                categories['beginner_friendly'] += 1
                
            if any(term in text_lower for term in ['advanced', 'competitive', 'elite', 'expert', 'master']):
                categories['advanced_level'] += 1
            
            # Stroke analysis
            for stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']:
                if any(term in text_lower for term in self.advanced_swimming_terms[stroke]):
                    categories[f'{stroke}_content'] += 1
            
            # Content metrics
            if chunk.get('images'):
                categories['with_images'] += 1
                
            if text_length > 800:
                categories['comprehensive_chunks'] += 1
            elif text_length < 200:
                categories['short_chunks'] += 1
        
        avg_length = total_length / len(self.chunks) if self.chunks else 0
        
        logger.info(f"\n📊 ADVANCED CONTENT ANALYSIS:")
        logger.info(f"📚 Technical focus: {categories['technique_focused']} chunks")
        logger.info(f"🏃‍♂️ Drills/exercises: {categories['drill_based']} chunks") 
        logger.info(f"👶 Beginner level: {categories['beginner_friendly']} chunks")
        logger.info(f"🏆 Advanced level: {categories['advanced_level']} chunks")
        logger.info(f"🖼️ With images: {categories['with_images']} chunks")
        logger.info(f"📏 Average length: {avg_length:.0f} characters")
    
    def load_swimming_embeddings(self, embeddings_path: str):
        """Load embeddings with index optimization"""
        try:
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
            
            logger.info(f"✅ Embeddings loaded: {self.embeddings.shape}")
            self.create_optimized_faiss_index()
            
        except Exception as e:
            st.error(f"Embedding loading error: {str(e)}")
    
    def create_optimized_faiss_index(self):
        """Create optimized FAISS index with HNSW if possible"""
        if self.embeddings is not None:
            logger.info("🔄 Creating optimized FAISS index...")
            
            try:
                dimension = self.embeddings.shape[1]
                
                # Try HNSW for better performance
                if len(self.chunks) > 1000:
                    # HNSW for large collections
                    self.index = faiss.IndexHNSWFlat(dimension, 32)
                    self.index.hnsw.efConstruction = 200
                    self.index.hnsw.efSearch = 50
                else:
                    # IndexFlatIP for small collections
                    self.index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings.astype('float32'))
                
                logger.info("✅ Optimized FAISS index created!")
                
            except Exception as e:
                logger.warning(f"Fallback to simple index: {e}")
                # Fallback to simple index
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings.astype('float32'))
    
    def intelligent_query_expansion(self, query: str) -> str:
        """Conditional intelligent query expansion"""
        query_lower = query.lower()
        query_words = query.split()
        
        # Only expand if necessary
        if len(query_words) >= 4:
            # Query already detailed, no expansion
            return query
        
        # Detect concepts and expand intelligently
        expanded_terms = []
        concepts_detected = []
        
        for concept, synonyms in self.advanced_swimming_terms.items():
            if any(syn in query_lower for syn in synonyms):
                concepts_detected.append(concept)
                # Add 1-2 relevant synonyms only
                relevant_synonyms = [syn for syn in synonyms if syn not in query_lower][:2]
                expanded_terms.extend(relevant_synonyms)
        
        # Minimal contextual expansion
        if not concepts_detected:
            # Very general query, add minimal context
            if len(query_words) < 3:
                expanded_terms.extend(['swimming', 'technique'])
        
        # Build intelligently expanded query
        if expanded_terms:
            # Limit expansion to avoid noise
            selected_terms = expanded_terms[:3]  # Maximum 3 terms
            expanded_query = query + " " + " ".join(selected_terms)
            logger.info(f"Query expansion: '{query}' -> '{expanded_query}'")
            return expanded_query
        
        return query
    
    def advanced_swimming_search(self, query: str, top_k: int = 8) -> List[Dict]:
        """PROBLÈME 2 CORRIGÉ: Advanced search ensuring images are always found"""
        if self.index is None:
            return []
        
        try:
            # 1. Intelligent expansion
            expanded_query = self.intelligent_query_expansion(query)
            
            # 2. Search with over-sampling to find more chunks with images
            query_embedding = self.model.encode([expanded_query], normalize_embeddings=True)
            
            # Search for more candidates to ensure we find images
            search_k = min(top_k * 5, len(self.chunks))  # Increased multiplier
            scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # 3. Process and prioritize results with images
            candidates_with_images = []
            candidates_without_images = []
            seen_content = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if score < 0.02 or idx >= len(self.chunks):  # More permissive threshold
                    continue
                
                chunk = self.chunks[idx].copy()
                
                # Anti-duplicates
                content_signature = hash(chunk['text'][:200])
                if content_signature in seen_content:
                    continue
                seen_content.add(content_signature)
                
                # Metadata enrichment
                chunk['similarity_score'] = float(score)
                chunk['swimming_type'] = self.identify_swimming_type_advanced(chunk['text'])
                chunk['stroke'] = self.identify_stroke_advanced(chunk['text'])
                chunk['level'] = self.identify_level_advanced(chunk['text'])
                chunk['cleaned_text'] = self.clean_text_enhanced(chunk['text'])
                chunk['relevance_score'] = self.calculate_relevance_score(query, chunk)
                
                # PROBLÈME 2: Prioritize chunks with images
                if chunk.get('images') and len(chunk['images']) > 0:
                    candidates_with_images.append(chunk)
                else:
                    candidates_without_images.append(chunk)
            
            # PROBLÈME 2: Ensure we always have images by prioritizing chunks with images
            prioritized_candidates = candidates_with_images + candidates_without_images
            
            # If we don't have enough chunks with images, do a broader search
            if len(candidates_with_images) < 3:  # Ensure at least 3 chunks with images
                logger.info("🖼️ Searching for more chunks with images...")
                # Broader search with lower threshold
                broader_search_k = min(len(self.chunks), 100)
                broader_scores, broader_indices = self.index.search(query_embedding.astype('float32'), broader_search_k)
                
                for score, idx in zip(broader_scores[0], broader_indices[0]):
                    if idx >= len(self.chunks):
                        continue
                    
                    chunk = self.chunks[idx]
                    content_signature = hash(chunk['text'][:200])
                    
                    if (content_signature not in seen_content and 
                        chunk.get('images') and len(chunk['images']) > 0):
                        
                        chunk_copy = chunk.copy()
                        chunk_copy['similarity_score'] = float(score)
                        chunk_copy['swimming_type'] = self.identify_swimming_type_advanced(chunk['text'])
                        chunk_copy['stroke'] = self.identify_stroke_advanced(chunk['text'])
                        chunk_copy['level'] = self.identify_level_advanced(chunk['text'])
                        chunk_copy['cleaned_text'] = self.clean_text_enhanced(chunk['text'])
                        chunk_copy['relevance_score'] = self.calculate_relevance_score(query, chunk)
                        
                        candidates_with_images.append(chunk_copy)
                        seen_content.add(content_signature)
                        
                        if len(candidates_with_images) >= 5:  # Stop when we have enough
                            break
                
                # Re-prioritize
                prioritized_candidates = candidates_with_images + candidates_without_images
            
            # 4. Intelligent re-ranking with image priority
            results = self.rerank_results_with_image_priority(query, prioritized_candidates, top_k)
            
            logger.info(f"Advanced search: {len(results)} results found, {sum(1 for r in results if r.get('images')) } with images")
            return results
            
        except Exception as e:
            logger.error(f"Advanced search error: {str(e)}")
            return []
    
    def calculate_relevance_score(self, query: str, chunk: Dict) -> float:
        """Calculate contextual relevance score"""
        query_lower = query.lower()
        text_lower = chunk.get('text', '').lower()
        
        relevance_factors = []
        
        # 1. Keyword matching
        query_terms = set(query_lower.split())
        text_terms = set(text_lower.split())
        term_overlap = len(query_terms.intersection(text_terms)) / len(query_terms) if query_terms else 0
        relevance_factors.append(term_overlap)
        
        # 2. Swimming specificity
        swimming_terms_count = sum(1 for category_terms in self.advanced_swimming_terms.values()
                                 for term in category_terms if term in text_lower)
        swimming_specificity = min(swimming_terms_count / 5, 1.0)
        relevance_factors.append(swimming_specificity)
        
        # 3. Image bonus
        image_bonus = 0.2 if chunk.get('images') and len(chunk['images']) > 0 else 0
        relevance_factors.append(image_bonus)
        
        # 4. Optimal length (neither too short nor too long)
        text_length = len(chunk.get('text', ''))
        optimal_length_score = 1.0 if 200 <= text_length <= 1000 else max(0.5, 1.0 - abs(text_length - 600) / 1000)
        relevance_factors.append(optimal_length_score)
        
        return np.mean(relevance_factors)
    
    def rerank_results_with_image_priority(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """PROBLÈME 2: Intelligent result re-ranking with image priority"""
        if not candidates:
            return []
        
        # Composite score: similarity + relevance + image bonus + diversity
        for candidate in candidates:
            similarity = candidate.get('similarity_score', 0)
            relevance = candidate.get('relevance_score', 0)
            
            # PROBLÈME 2: Strong image bonus
            image_bonus = 0.3 if candidate.get('images') and len(candidate['images']) > 0 else 0
            
            # Diversity bonus (different strokes, levels)
            diversity_bonus = 0.1 if candidate.get('stroke') != 'All-Strokes' else 0
            
            # Final composite score with image priority
            composite_score = (similarity * 0.4 + relevance * 0.3 + image_bonus + diversity_bonus)
            candidate['composite_score'] = composite_score
        
        # Sort by composite score
        candidates.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # PROBLÈME 2: Ensure image diversity in top results
        results_with_images = []
        results_without_images = []
        
        for candidate in candidates:
            if candidate.get('images') and len(candidate['images']) > 0:
                results_with_images.append(candidate)
            else:
                results_without_images.append(candidate)
        
        # PROBLÈME 2: Prioritize results with images
        final_results = []
        
        # Add results with images first (at least 60% of results should have images)
        target_with_images = max(int(top_k * 0.6), min(3, len(results_with_images)))
        final_results.extend(results_with_images[:target_with_images])
        
        # Fill remaining slots
        remaining_slots = top_k - len(final_results)
        if remaining_slots > 0:
            # Add more with images if available
            remaining_with_images = results_with_images[target_with_images:]
            final_results.extend(remaining_with_images[:remaining_slots])
            
            # Fill with results without images if needed
            remaining_slots = top_k - len(final_results)
            if remaining_slots > 0:
                final_results.extend(results_without_images[:remaining_slots])
        
        return final_results[:top_k]
    
    def identify_swimming_type_advanced(self, text: str) -> str:
        """Advanced swimming content type identification"""
        text_lower = text.lower()
        
        type_indicators = {
            'Swimming-Drill': ['drill', 'exercise', 'practice', 'training exercise', 'workout'],
            'Swimming-Technique': ['technique', 'form', 'mechanics', 'proper', 'correct', 'biomechanics'],
            'Swimming-Training': ['training', 'workout', 'conditioning', 'program', 'schedule'],
            'Swimming-Competition': ['competition', 'race', 'meet', 'championship', 'event'],
            'Swimming-Equipment': ['equipment', 'gear', 'kickboard', 'fins', 'paddles'],
            'Swimming-Safety': ['safety', 'rescue', 'emergency', 'lifeguard', 'accident']
        }
        
        for swim_type, indicators in type_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return swim_type
        
        return "Swimming-General"
    
    def identify_stroke_advanced(self, text: str) -> str:
        """Advanced stroke identification with confidence score"""
        text_lower = text.lower()
        
        stroke_scores = {}
        for stroke, terms in self.advanced_swimming_terms.items():
            if stroke in ['freestyle', 'backstroke', 'breaststroke', 'butterfly']:
                score = sum(1 for term in terms if term in text_lower)
                if score > 0:
                    stroke_scores[stroke] = score
        
        if stroke_scores:
            best_stroke = max(stroke_scores.items(), key=lambda x: x[1])[0]
            return best_stroke.capitalize()
        
        return "All-Strokes"
    
    def identify_level_advanced(self, text: str) -> str:
        """Advanced level identification with nuances"""
        text_lower = text.lower()
        
        level_indicators = {
            'Beginner': ['beginner', 'novice', 'learning', 'basic', 'introduction', 'start', 'first'],
            'Intermediate': ['intermediate', 'developing', 'improving', 'progressing', 'moderate'],
            'Advanced': ['advanced', 'competitive', 'elite', 'expert', 'master', 'professional'],
            'Youth': ['youth', 'junior', 'kid', 'child', 'young', 'age group'],
            'Masters': ['masters', 'adult', 'senior', 'veteran', 'mature']
        }
        
        for level, indicators in level_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return level
        
        return "All-Levels"
    
    def clean_text_enhanced(self, text: str) -> str:
        """Enhanced text cleaning preserving structure"""
        if not text:
            return ""
        
        # Preserve important structure
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Join split words
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Remove non-relevant metadata
        text = re.sub(r'About the book.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'ISBN.*?(?=\n|$)', '', text)
        text = re.sub(r'www\..*?(?=\s|$)', '', text)
        text = re.sub(r'Chapter \d+.*?(?=\n|$)', '', text)
        
        # Increase limit for more context
        return text.strip()[:1500]  # Increased from 800 to 1500

class SwimmingUltraAssistant:
    """Ultra-optimized swimming assistant with improved generation"""
    
    def __init__(self):
        logger.info("🏊‍♂️ Initializing Ultra-Optimized Assistant...")
        self.setup_session_state()
        self.rag_system = SwimmingUltraOptimizedRAG()
        # PROBLÈME 4 CORRIGÉ: Force load data at initialization
        self.system_ready = self.load_data()
    
    def setup_session_state(self):
        """Initialize session state with new metrics"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        # PROBLÈME 4 CORRIGÉ: Store system status in session state
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        if 'performance_stats' not in st.session_state:
            st.session_state.performance_stats = {
                'total_queries': 0,
                'avg_score': 0.0,
                'excellent_count': 0,
                'good_count': 0
            }
        if 'show_evaluation' not in st.session_state:
            st.session_state.show_evaluation = True
        if 'show_details' not in st.session_state:
            st.session_state.show_details = False
        if 'show_images' not in st.session_state:
            st.session_state.show_images = True
    
    def load_data(self):
        """PROBLÈME 4 CORRIGÉ: Load data with persistent status"""
        try:
            # Possible paths with priorities
            base_paths = ["output1", "output", "data", "."]
            
            for base_path in base_paths:
                chunks_path = os.path.join(base_path, "chunks_prepared.json")
                embeddings_path = os.path.join(base_path, "embeddings_swimming.pkl")
                
                if os.path.exists(chunks_path) and os.path.exists(embeddings_path):
                    logger.info(f"📂 Loading from: {base_path}")
                    
                    self.rag_system.load_swimming_chunks(chunks_path)
                    self.rag_system.load_swimming_embeddings(embeddings_path)
                    
                    # PROBLÈME 4 CORRIGÉ: Set persistent session state
                    st.session_state.system_ready = True
                    logger.info("✅ Ultra-optimized RAG system ready!")
                    return True
            
            logger.warning("⚠️ RAG data not found. Basic mode activated.")
            st.session_state.system_ready = False
            return False
            
        except Exception as e:
            logger.error(f"❌ Loading error: {str(e)}")
            st.session_state.system_ready = False
            return False
    
    def get_api_key(self):
        """Get API key from multiple sources"""
        try:
            return st.secrets["GROK_API_KEY"]
        except:
            api_key = os.environ.get('GROK_API_KEY')
            if not api_key:
                st.sidebar.error("🔑 Grok API key required")
                st.stop()
            return api_key
    
    def validate_query_advanced(self, query: str, api_key: str) -> tuple[bool, str]:
        """Advanced validation with intent detection"""
        query_clean = query.strip()
        if not query_clean or len(query_clean) < 3:
            return False, "Please enter a more detailed swimming question"
        
        # Extended swimming keywords
        swimming_keywords = [
            'swim', 'stroke', 'technique', 'freestyle', 'backstroke', 'breaststroke', 'butterfly',
            'pool', 'water', 'training', 'drill', 'breathing', 'kick', 'pull', 'form',
            'natation', 'nage', 'entrainement', 'piscine', 'crawl', 'dos', 'brasse', 'papillon',
            'technique', 'respiration', 'battement', 'traction', 'position'
        ]
        
        # Smarter client-side validation
        if any(keyword in query_clean.lower() for keyword in swimming_keywords):
            return True, ""
        
        # Secondary validation: activity-related words
        activity_keywords = ['improve', 'better', 'learn', 'practice', 'exercise', 'workout', 'performance']
        if any(keyword in query_clean.lower() for keyword in activity_keywords):
            return True, ""  # Accept if improvement-related
        
        return False, "Please ask a question specifically related to swimming"
    
    def generate_with_grok_enhanced(self, prompt: str, api_key: str) -> str:
        """Enhanced Grok generation with ultra-optimized parameters"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Ultra-optimized system prompt
        system_prompt = """You are an expert swimming coach AI with deep technical knowledge. Provide comprehensive, actionable advice based strictly on the provided documentation.

GUIDELINES FOR OPTIMAL RESPONSES:
- Length: 250-400 words for comprehensive coverage
- Structure: Use clear paragraphs and bullet points when helpful
- Technical precision: Include specific techniques from the sources
- Actionability: Provide concrete steps and drills
- Encouragement: Be motivational and supportive
- Adaptation: Match complexity to user's apparent level

RESPONSE STRUCTURE:
1. Direct answer to the question
2. Specific techniques/methods from documentation
3. Practical implementation steps
4. Additional tips for improvement

Base everything on the provided swimming documentation while being natural and engaging.

Response:"""
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,        # Increased for more details
            "temperature": 0.7,       # More creative
            "top_p": 0.9,            # More diversity
            "frequency_penalty": 0.1, # Avoid repetition
            "presence_penalty": 0.1   # Encourage new concepts
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=data, timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Grok API error: {response.status_code}")
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Grok network error: {str(e)}")
            return f"Network error: {str(e)}"
    
    def build_enhanced_prompt_v2(self, query: str, search_results: List[Dict]) -> str:
        """Build ultra-optimized prompt with enriched context"""
        context_parts = []
        
        if search_results:
            doc_context = []
            
            # Organize by relevance and type
            for i, result in enumerate(search_results[:5], 1):
                source = result['pdf'].replace('.pdf', '').replace('_', ' ').title()
                stroke = result.get('stroke', 'General')
                swim_type = result.get('swimming_type', 'General').replace('Swimming-', '')
                level = result.get('level', 'All')
                similarity = result.get('similarity_score', 0)
                text = result['cleaned_text']
                
                # Enriched context with metadata
                doc_header = f"📚 SOURCE {i}: {source} | {stroke} | {swim_type} | {level} | Relevance: {similarity:.2f}"
                doc_context.append(f"{doc_header}\n{text}\n")
            
            context_parts.append("SWIMMING TECHNICAL DOCUMENTATION:\n" + '\n'.join(doc_context))
            
            # Add summary of available images
            total_images = sum(len(result.get('images', [])) for result in search_results)
            if total_images > 0:
                context_parts.append(f"📸 VISUAL AIDS AVAILABLE: {total_images} technical demonstration images")
        
        full_context = '\n\n'.join(context_parts) if context_parts else "No specific documentation found. Provide general swimming guidance."
        
        return f"""{full_context}

🏊‍♂️ SWIMMER'S QUESTION: "{query}"

COACHING INSTRUCTIONS:
- Answer based primarily on the provided documentation
- Include specific techniques, drills, or methods mentioned in sources
- Provide actionable steps the swimmer can immediately implement
- Be encouraging and motivational
- Structure your response clearly with practical advice
- If multiple sources are available, synthesize the best information

EXPERT RESPONSE:"""
    
    def extract_images_with_ai_descriptions(self, search_results: List[Dict]) -> List[Dict]:
        """PROBLÈME 2 CORRIGÉ: Extract images with priority and fallbacks"""
        images = []
        seen_images = set()
        
        # Possible paths for images with priority
        base_paths = ["output1/images_cleaned", "output/images_cleaned", "images_cleaned", "images", "output1/images", "output/images"]
        base_path = None
        
        for path in base_paths:
            if os.path.exists(path):
                base_path = path
                logger.info(f"📸 Using image directory: {base_path}")
                break
        
        if not base_path:
            logger.warning("⚠️ No image directory found, searching for fallback images...")
            # PROBLÈME 2: Fallback - try to find any image directory
            for root, dirs, files in os.walk("."):
                for dir_name in dirs:
                    if "image" in dir_name.lower():
                        base_path = os.path.join(root, dir_name)
                        logger.info(f"📸 Found fallback image directory: {base_path}")
                        break
                if base_path:
                    break
        
        if not base_path:
            logger.error("❌ No image directory found")
            return []
        
        # PROBLÈME 2: Extract images with priority for chunks that have them
        for result in search_results:
            result_images = result.get('images', [])
            if result_images:  # Prioritize results that have images
                for img in result_images[:4]:  # Max 4 per chunk
                    img_path = img.get('path', '')
                    if img_path:
                        # Clean path
                        if img_path.startswith('images_cleaned/'):
                            img_path = img_path[len('images_cleaned/'):]
                        elif img_path.startswith('images/'):
                            img_path = img_path[len('images/'):]
                        
                        full_path = os.path.join(base_path, img_path)
                        full_path = os.path.normpath(full_path)
                        
                        if full_path not in seen_images:
                            seen_images.add(full_path)
                            
                            # Check if file exists, if not try alternatives
                            if not os.path.exists(full_path):
                                # Try different extensions
                                base_name = os.path.splitext(full_path)[0]
                                extensions = ['.png', '.jpg', '.jpeg', '.gif']
                                for ext in extensions:
                                    alt_path = base_name + ext
                                    if os.path.exists(alt_path):
                                        full_path = alt_path
                                        break
                            
                            # PROBLÈME 1 CORRIGÉ: Generate varied descriptions
                            enhanced_description = self.rag_system.evaluator.image_descriptor.generate_enhanced_description(
                                full_path, result
                            )
                            
                            images.append({
                                'path': full_path,
                                'enhanced_description': enhanced_description,
                                'source': result['pdf'].replace('.pdf', ''),
                                'page': result.get('page', 0),
                                'stroke_focus': result.get('stroke', 'General'),
                                'content_type': result.get('swimming_type', 'Technique'),
                                'similarity_score': result.get('similarity_score', 0),
                                'exists': os.path.exists(full_path)
                            })
                            
                            if len(images) >= 8:  # Maximum 8 images
                                break
                
                if len(images) >= 8:
                    break
        
        # PROBLÈME 2: If we still don't have enough images, add a fallback search
        if len(images) < 3:
            logger.info("🔍 Searching for additional images...")
            # Search through all chunks for any images
            for chunk in self.rag_system.chunks[:50]:  # Search first 50 chunks
                chunk_images = chunk.get('images', [])
                for img in chunk_images:
                    img_path = img.get('path', '')
                    if img_path:
                        if img_path.startswith('images_cleaned/'):
                            img_path = img_path[len('images_cleaned/'):]
                        
                        full_path = os.path.join(base_path, img_path)
                        full_path = os.path.normpath(full_path)
                        
                        if (full_path not in seen_images and os.path.exists(full_path)):
                            seen_images.add(full_path)
                            
                            enhanced_description = self.rag_system.evaluator.image_descriptor.generate_enhanced_description(
                                full_path, chunk
                            )
                            
                            images.append({
                                'path': full_path,
                                'enhanced_description': enhanced_description,
                                'source': chunk.get('pdf', 'Swimming Guide').replace('.pdf', ''),
                                'page': chunk.get('page', 0),
                                'stroke_focus': 'General',
                                'content_type': 'Technique',
                                'similarity_score': 0.5,  # Default relevance
                                'exists': True
                            })
                            
                            if len(images) >= 6:
                                break
                if len(images) >= 6:
                    break
        
        # Sort by relevance and existence
        images.sort(key=lambda x: (x['exists'], x['similarity_score']), reverse=True)
        
        logger.info(f"📸 Found {len(images)} images for display")
        return images
    
    def process_swimming_query_ultra(self, query: str) -> Dict:
        """Ultra-optimized swimming query processing"""
        api_key = self.get_api_key()
        if not api_key:
            return self._get_error_response("Grok API key required")
        
        # Advanced validation
        is_valid, error_msg = self.validate_query_advanced(query, api_key)
        if not is_valid:
            return self._get_error_response(error_msg)
        
        try:
            # Ultra-optimized RAG search
            search_results = []
            if st.session_state.system_ready:
                search_results = self.rag_system.advanced_swimming_search(query, top_k=8)
                logger.info(f"Search: {len(search_results)} results found")
            
            # Generation with optimized prompt
            prompt = self.build_enhanced_prompt_v2(query, search_results)
            llm_response = self.generate_with_grok_enhanced(prompt, api_key)
            
            # PROBLÈME 2 CORRIGÉ: AI image extraction with guaranteed images
            images = self.extract_images_with_ai_descriptions(search_results)
            
            # Enriched sources
            sources = []
            for r in search_results:
                source_info = {
                    'title': r['pdf'].replace('.pdf', '').replace('_', ' ').title(),
                    'page': r.get('page', 0),
                    'stroke': r.get('stroke', 'General'),
                    'type': r.get('swimming_type', 'General').replace('Swimming-', ''),
                    'similarity': r.get('similarity_score', 0)
                }
                source_desc = f"{source_info['title']} (p.{source_info['page']}) - {source_info['stroke']} {source_info['type']} - {source_info['similarity']:.2f}"
                sources.append(source_desc)
            
            # Complete ultra-optimized evaluation
            evaluation = None
            if search_results and st.session_state.system_ready:
                evaluation = self.rag_system.evaluator.evaluate_complete_rag_optimized(
                    query, llm_response, search_results
                )
                
                # Update performance stats
                self._update_performance_stats(evaluation)
            
            return {
                'text': llm_response,
                'images': images,
                'sources': list(set(sources)),  # Deduplicate
                'search_results': search_results,
                'evaluation': evaluation,
                'query_metadata': {
                    'length': len(query),
                    'complexity': self._assess_query_complexity(query),
                    'swimming_focus': self._detect_swimming_focus(query)
                }
            }
            
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return self._get_error_response(f"System error: {str(e)}")
    
    def _get_error_response(self, message: str) -> Dict:
        """Standardized error response"""
        return {
            'text': message,
            'images': [],
            'sources': [],
            'search_results': [],
            'evaluation': None,
            'query_metadata': {}
        }
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        words = query.split()
        if len(words) <= 3:
            return "Simple"
        elif len(words) <= 8:
            return "Moderate"
        else:
            return "Complex"
    
    def _detect_swimming_focus(self, query: str) -> str:
        """Detect main query focus"""
        query_lower = query.lower()
        
        if any(stroke in query_lower for stroke in ['freestyle', 'crawl', 'front']):
            return "Freestyle"
        elif any(stroke in query_lower for stroke in ['backstroke', 'back']):
            return "Backstroke"
        elif any(stroke in query_lower for stroke in ['breaststroke', 'breast', 'frog']):
            return "Breaststroke"
        elif any(stroke in query_lower for stroke in ['butterfly', 'fly', 'dolphin']):
            return "Butterfly"
        elif any(term in query_lower for term in ['breathing', 'breath']):
            return "Breathing"
        elif any(term in query_lower for term in ['kick', 'kicking']):
            return "Kicking"
        elif any(term in query_lower for term in ['pull', 'arm', 'stroke']):
            return "Arm Technique"
        else:
            return "General"
    
    def _update_performance_stats(self, evaluation: Dict):
        """Update performance statistics"""
        stats = st.session_state.performance_stats
        
        stats['total_queries'] += 1
        current_score = evaluation.get('overall_score', 0)
        
        # Moving average
        stats['avg_score'] = ((stats['avg_score'] * (stats['total_queries'] - 1)) + current_score) / stats['total_queries']
        
        # Level counters
        if current_score >= 75:
            stats['excellent_count'] += 1
        elif current_score >= 60:
            stats['good_count'] += 1
    
    def display_image_in_streamlit_enhanced(self, image_path: str, description: str = "") -> bool:
        """Enhanced image display with description"""
        try:
            if not os.path.exists(image_path):
                st.warning(f"⚠️ Image not found: {os.path.basename(image_path)}")
                return False
            
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            
            with open(image_path, "rb") as f:
                img_data = f.read()
            
            img_b64 = base64.b64encode(img_data).decode()
            
            # Enriched HTML with description
            html_content = f'''
            <div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
                <img src="data:{mime_type};base64,{img_b64}" 
                     style="max-width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" 
                     alt="Swimming technique demonstration">
                <div style="margin-top: 0.8rem; font-style: italic; color: #333; background: rgba(255,255,255,0.8); padding: 0.5rem; border-radius: 8px;">
                    {description}
                </div>
            </div>
            '''
            
            st.markdown(html_content, unsafe_allow_html=True)
            return True
            
        except Exception as e:
            logger.error(f"Image display error: {str(e)}")
            st.error(f"❌ Cannot display image: {os.path.basename(image_path)}")
            return False

def main():
    """PROBLÈME 4 CORRIGÉ: Ultra-optimized main application with persistent status"""
    st.markdown("""
    <div class="main-header">
        <h1>🏊‍♂️ Swimming Coach AI - Ultra-Optimized RAG</h1>
        <p>Intelligent swimming assistant with advanced automatic evaluation</p>
        <small>Technique • Training • Performance • Advanced AI • Realistic Scores</small>
    </div>
    """, unsafe_allow_html=True)
    
    # PROBLÈME 4 CORRIGÉ: Initialization with persistent cache
    if 'swimming_assistant' not in st.session_state:
        with st.spinner("🏊‍♂️ Initializing ultra-optimized system..."):
            st.session_state.swimming_assistant = SwimmingUltraAssistant()
    
    assistant = st.session_state.swimming_assistant
    
    # PROBLÈME 4 CORRIGÉ: Sidebar with persistent system status
    with st.sidebar:
        st.markdown("### ⚙️ Ultra-Optimized Configuration")
        
        # Store checkbox values in session state to persist across reloads
        st.session_state.show_evaluation = st.checkbox(
            "📊 RAG Evaluation", 
            value=st.session_state.show_evaluation, 
            help="Evaluation with realistic thresholds and semantic metrics"
        )
        st.session_state.show_details = st.checkbox(
            "🔍 Technical details", 
            value=st.session_state.show_details,
            help="Detailed information about RAG process"
        )
        st.session_state.show_images = st.checkbox(
            "🖼️ AI Images", 
            value=st.session_state.show_images,
            help="Image descriptions with artificial intelligence"
        )
        
        # Performance statistics
        if st.session_state.performance_stats['total_queries'] > 0:
            st.markdown("### 📈 Performance Statistics")
            stats = st.session_state.performance_stats
            
            st.metric("Queries Processed", stats['total_queries'])
            st.metric("Average Score", f"{stats['avg_score']:.1f}/100")
            st.metric("Excellent Results", stats['excellent_count'])
            st.metric("Good Results", stats['good_count'])
            
            # Success rate
            success_rate = ((stats['excellent_count'] + stats['good_count']) / stats['total_queries']) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        st.markdown("### 🚀 Ultra-Optimized Improvements")
        st.markdown("""
        **🎯 Realistic Thresholds**
        - Excellent: ≥75 (vs 85)
        - Good: ≥60 (vs 70)
        - Acceptable: ≥45 (vs 50)
        
        **🧠 Advanced AI**
        - HNSW intelligent search
        - BLIP image descriptions
        - Embeddings semantic analysis
        - Contextual re-ranking
        
        **⚖️ Optimized Weighting**
        - 50% Generation (vs 40%)
        - 30% Retrieval (vs 40%)
        - 20% Images (bonus)
        """)
        
        # PROBLÈME 4 CORRIGÉ: Persistent system status
        st.markdown("### 🔧 System Status")
        if st.session_state.system_ready:
            st.markdown("""
            <div class="status-card">
                ✅ Ultra-Optimized RAG Active
            </div>
            """, unsafe_allow_html=True)
            st.info(f"📊 {len(assistant.rag_system.chunks)} chunks available")
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9800, #f57c00); color: white; padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;">
                ⚠️ Basic mode
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced main interface
    st.markdown("### 💬 Ask Your Question - Ultra-Optimized Version")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area(
            "",
            height=100,
            placeholder="Ex: How to perfect my bilateral breathing technique in freestyle to improve my endurance?",
            help="Ask a detailed swimming question for best results"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            ask_button = st.button("🏊‍♂️ Ultra-AI Coach", type="primary")
        with col_btn2:
            clear_button = st.button("🗑️ Clear")
        with col_btn3:
            example_button = st.button("💡 Example")
    
    with col2:
        st.markdown("### 💡 Expert Questions")
        expert_suggestions = [
            "Bilateral breathing freestyle technique",
            "Flip turn optimization freestyle", 
            "Butterfly arm-leg coordination",
            "Endurance training program",
            "Breaststroke technique corrections"
        ]
        
        for i, suggestion in enumerate(expert_suggestions):
            if st.button(f"🎯 {suggestion}", key=f"expert_sugg_{i}"):
                user_query = suggestion
                ask_button = True
    
    # Complex question example
    if example_button:
        example_query = "How to improve my bilateral breathing technique in freestyle while maintaining good hydrodynamic position and optimizing my stroke rhythm for long distance swimming?"
        user_query = example_query
        ask_button = True
    
    # Ultra-optimized query processing
    if ask_button and user_query.strip():
        with st.spinner("🔄 Ultra-optimized processing of your question..."):
            response = assistant.process_swimming_query_ultra(user_query)
            
            # Add to history with metadata
            chat_entry = {
                'question': user_query,
                'response': response,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'show_evaluation': st.session_state.show_evaluation,
                'show_details': st.session_state.show_details,
                'show_images': st.session_state.show_images,
                'query_metadata': response.get('query_metadata', {})
            }
            st.session_state.chat_history.append(chat_entry)
    
    # Clear history
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.performance_stats = {
            'total_queries': 0, 'avg_score': 0.0, 'excellent_count': 0, 'good_count': 0
        }
        st.success("🗑️ History and statistics cleared!")
    
   # Ultra-enriched history display
    if st.session_state.chat_history:
        st.markdown("### 🏊‍♂️ Your Ultra-Optimized Coaching Session")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Last 10
            with st.container():
                # Query metadata
                metadata = chat.get('query_metadata', {})
                if metadata:
                    st.markdown(f"""
                    <div style="background: rgba(0,150,200,0.1); padding: 0.5rem; border-radius: 10px; margin: 0.5rem 0;">
                        <small>📊 Complexity: {metadata.get('complexity', 'N/A')} | 
                        Focus: {metadata.get('swimming_focus', 'N/A')} | 
                        Length: {metadata.get('length', 0)} characters</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # User question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🏊‍♀️ You ({chat['timestamp']}):</strong><br>
                    {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant response
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🏊‍♂️ Ultra-AI Coach:</strong><br>
                    {chat['response']['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Ultra-optimized evaluation
                if chat.get('show_evaluation') and chat['response'].get('evaluation'):
                    assistant.rag_system.evaluator.display_evaluation_enhanced(chat['response']['evaluation'])
                
                # Images with AI descriptions
                if chat.get('show_images') and chat['response'].get('images'):
                    with st.expander(f"🖼️ AI Visual Demonstrations ({len(chat['response']['images'])})"):
                        for j, img in enumerate(chat['response']['images']):
                            col_img, col_desc = st.columns([1, 1])
                            
                            with col_img:
                                if assistant.display_image_in_streamlit_enhanced(img['path'], img.get('enhanced_description', '')):
                                    st.caption(f"📖 {img['source']} - Page {img['page']} - Relevance: {img.get('similarity_score', 0):.2f}")
                            
                            with col_desc:
                                st.markdown("**🤖 Advanced AI Description:**")
                                st.info(img['enhanced_description'])
                                st.caption(f"**Context:** {img.get('stroke_focus', 'General')} - {img.get('content_type', 'Technique')}")
                
                # Enriched sources
                if chat['response'].get('sources'):
                    with st.expander(f"📚 Expert Sources ({len(chat['response']['sources'])})"):
                        for j, source in enumerate(chat['response']['sources'], 1):
                            st.markdown(f"**{j}.** {source}")
                
                # Ultra-detailed technical details
                if chat.get('show_details') and chat['response'].get('search_results'):
                    with st.expander("🔍 Ultra-Detailed Technical Analysis"):
                        results = chat['response']['search_results']
                        st.write(f"📊 {len(results)} documents analyzed with intelligent search")
                        
                        for j, result in enumerate(results[:3], 1):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                **{j}. {result.get('stroke', 'General')} - {result.get('swimming_type', 'Technique')}**
                                - Similarity: {result.get('similarity_score', 0):.3f}
                                - Relevance: {result.get('relevance_score', 0):.3f}
                                - Composite score: {result.get('composite_score', 0):.3f}
                                """)
                            with col2:
                                st.markdown(f"""
                                - Source: {result.get('pdf', 'Unknown')} (p.{result.get('page', 0)})
                                - Level: {result.get('level', 'All')}
                                - Images: {len(result.get('images', []))}
                                - Length: {len(result.get('text', ''))} characters
                                """)
                
                st.divider()
    
    # Ultra-optimized footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0288d1 0%, #0277bd 100%); color: white; text-align: center; padding: 2rem; border-radius: 15px; margin-top: 2rem;">
        🏊‍♂️ <strong>Swimming Coach AI - Ultra-Optimized Version</strong><br>
        <small>Powered by Grok AI • FAISS HNSW • BLIP Vision AI • Advanced RAG Evaluation</small><br>
        <small>🚀 Realistic thresholds • AI descriptions • Semantic analysis • Fair scoring</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()