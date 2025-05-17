# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, Field, ValidationError
from markupsafe import escape
import os
import requests
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

app = Flask(__name__)

# ====================
# CORS Configuration
# ====================
allowed_origins = [
    "https://trish-a-gen-ai-chatbot.onrender.com",  
    "http://localhost:3000"                    # For local testing
]
CORS(app, resources={
    r"/api/*": {
        "origins": allowed_origins,
        "supports_credentials": True
    }
})

# ====================
# App Configuration
# ====================
app.config.update({
    'JSON_SORT_KEYS': False,
    'RATELIMIT_HEADERS_ENABLED': True,
    'ENV': 'production' if os.environ.get('ENV') == 'production' else 'development'
})

# ====================
# Rate Limiting
# ====================
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ====================
# Logging Setup
# ====================
handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024 * 1024,  # 1MB per file
    backupCount=3
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# ====================
# Environment Variables
# ====================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("AI_MODEL", "meta-llama/llama-3-70b-instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

# ====================
# Data Models
# ====================
class ChatRequest(BaseModel):
    messages: list = Field(..., min_items=1)
    discussionTitle: str = Field(..., min_length=2, max_length=100)
    user_id: Optional[str] = Field(None, min_length=1)

class ConclusionRequest(ChatRequest):
    focus_areas: Optional[list] = Field(None, min_items=1)

# ====================
# Helper Functions
# ====================
def sanitize_input(text: str) -> str:
    return escape(text.strip())

def create_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": os.environ.get("REFERER_URL", "https://your-domain.com"),
        "X-Title": "Discussion Facilitator AI"
    }

# ====================
# AI Service Handler
# ====================
def get_ai_response(messages: list, discussion_context: str, system_prompt: Optional[str] = None) -> str:
    try:
        sanitized_context = sanitize_input(discussion_context)
        
        system_prompt = system_prompt or f"""You are TRISH, an AI discussion facilitator. Guide discussion about: {sanitized_context}.
        Role:
        1. Ask thought-provoking questions
        2. Provide relevant insights
        3. Ensure balanced participation
        4. Maintain focus on topic
        Keep responses concise (1-2 paragraphs)."""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=create_headers(),
            json=payload,
            timeout=15
        )
        response.raise_for_status()

        response_data = response.json()
        if not response_data.get("choices"):
            raise ValueError("Invalid AI response structure")
        
        content = response_data["choices"][0]["message"].get("content", "")
        return content if content.strip() else "Response unavailable"

    except requests.exceptions.HTTPError as http_err:
        app.logger.error(f"HTTP Error: {http_err}")
        return "Service temporarily unavailable. Please try again."
    except Exception as e:
        app.logger.error(f"AI Service Error: {str(e)}")
        return "Discussion service is currently unavailable."

# ====================
# API Endpoints
# ====================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.2.0",
        "environment": app.config['ENV']
    })

@app.route('/api/chat', methods=['POST'])
@limiter.limit("15/minute")
def chat():
    try:
        data = ChatRequest(**request.json).dict()
        response = get_ai_response(
            messages=data['messages'],
            discussion_context=data['discussionTitle']
        )
        return jsonify({"response": response})
    except ValidationError as e:
        return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        app.logger.error(f"Chat Error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

@app.route('/api/generate-conclusion', methods=['POST'])
@limiter.limit("10/minute")
def generate_conclusion():
    try:
        data = ConclusionRequest(**request.json).dict()
        conclusion_prompt = f"""Generate conclusion for: {sanitize_input(data['discussionTitle'])}.
        Format in Markdown with:
        ## Key Points
        ## Insights
        ## Agreements
        ## Recommendations"""
        
        response = get_ai_response(
            messages=data['messages'],
            discussion_context=data['discussionTitle'],
            system_prompt=conclusion_prompt
        )
        return jsonify({"conclusion": response})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Conclusion generation failed"}), 500

# ====================
# Error Handlers
# ====================
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too many requests",
        "message": "Please wait before making new requests"
    }), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Endpoint not found"}), 404

# ====================
# Production Setup
# ====================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
