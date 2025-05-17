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
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Configuration
app.config.update({
    'JSON_SORT_KEYS': False,
    'RATELIMIT_HEADERS_ENABLED': True
})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Logging setup
handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024 * 1024,
    backupCount=3
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("AI_MODEL", "meta-llama/llama-3-70b-instruct")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

CONCLUSION_PROMPT = """Generate comprehensive conclusion with EXACT structure:

## Key Points Summary
- [3-5 main points as bullet points]

## Major Insights
- [Notable observations from discussion]

## Agreements Reached
- [Clear consensus items]

## Action Plan
- [Concrete next steps with owners/dates]

EXAMPLE:
## Key Points Summary
- Team aligned on Q3 milestones
- Budget concerns about cloud costs
- UX design approval pending

Discussion: {discussion_title}

INSTRUCTIONS:
1. Use exactly 4 sections
2. Minimum 3 bullets per section
3. Follow example format
4. Be specific
5. Maintain professional tone
"""

# Pydantic models
class ChatRequest(BaseModel):
    messages: list = Field(..., min_items=1)
    discussionTitle: str = Field(..., min_length=2, max_length=100)
    user_id: Optional[str] = Field(None, min_length=1)

class ConclusionRequest(ChatRequest):
    focus_areas: Optional[list] = Field(None, min_items=1)

# Helper functions
def sanitize_input(text: str) -> str:
    return escape(text.strip())

def create_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": os.getenv("REFERER_URL", "https://your-domain.com"),
        "X-Title": "Discussion Facilitator AI"
    }

def get_ai_response(messages: list, discussion_context: str, system_prompt: Optional[str] = None) -> str:
    try:
        sanitized_context = sanitize_input(discussion_context)
        
        if system_prompt:
            final_prompt = CONCLUSION_PROMPT.format(discussion_title=sanitized_context)
            temp = 0.3  # Lower temperature for structured output
            max_tokens = 1200
        else:
            final_prompt = f"""You are TRISH, an AI discussion facilitator. Current discussion: {sanitized_context}.
            Your role:
            - Ask probing questions
            - Identify common ground
            - Challenge assumptions
            - Manage timekeeping
            - Ensure participation balance"""
            temp = TEMPERATURE
            max_tokens = MAX_TOKENS

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": final_prompt},
                *messages
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
            "top_p": 0.7,
            "frequency_penalty": 0.7 if system_prompt else 0.5,
            "presence_penalty": 0.4 if system_prompt else 0.3
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
            raise ValueError("Invalid response structure from AI provider")
        
        content = response_data["choices"][0]["message"].get("content", "")
        if not content.strip():
            raise ValueError("Empty response content")
            
        return content

    except requests.exceptions.HTTPError as http_err:
        app.logger.error(f"HTTP Error: {http_err}")
        return "Our AI service is currently experiencing high demand. Please try again in a moment."
    except (KeyError, IndexError) as key_err:
        app.logger.error(f"Response parsing error: {key_err}")
        return "We're having trouble processing the AI response. Please try again."
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return "Our discussion service is temporarily unavailable. Please try again later."

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": "1.3.0"})

@app.route('/api/chat', methods=['POST'])
@limiter.limit("15/minute")
def chat():
    try:
        req = ChatRequest(**request.json)
        data = req.dict()
    except ValidationError as e:
        app.logger.warning(f"Validation error: {str(e)}")
        return jsonify({"error": "Invalid request format"}), 400

    try:
        response = get_ai_response(
            messages=data['messages'],
            discussion_context=data['discussionTitle']
        )
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/generate-conclusion', methods=['POST'])
@limiter.limit("10/minute")
def generate_conclusion():
    try:
        req = ConclusionRequest(**request.json)
        data = req.dict()
        
        if len(data['messages']) < 3:
            return jsonify({
                "error": "Not enough discussion content to generate conclusion",
                "min_messages": 3,
                "retry": False
            }), 400

        response = get_ai_response(
            messages=data['messages'],
            discussion_context=data['discussionTitle'],
            system_prompt=CONCLUSION_PROMPT
        )
        
        # Validate conclusion structure
        required_sections = [
            'Key Points Summary',
            'Major Insights',
            'Agreements Reached',
            'Action Plan'
        ]
        if not all(section in response for section in required_sections):
            raise ValueError("Generated conclusion missing required sections")
            
        return jsonify({"conclusion": response})
        
    except ValidationError as e:
        return jsonify({"error": str(e), "retry": False}), 400
    except ValueError as ve:
        app.logger.error(f"Conclusion validation failed: {str(ve)}")
        return jsonify({
            "error": "Conclusion generation quality check failed",
            "message": str(ve),
            "retry": True
        }), 500
    except Exception as e:
        app.logger.error(f"Conclusion error: {str(e)}\nMessages: {data.get('messages', [])}")
        return jsonify({
            "error": "Failed to generate conclusion",
            "message": str(e),
            "retry": True
        }), 500

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Too many requests",
        "message": "Please slow down your requests",
        "retry": True
    }), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Endpoint not found", "retry": False}), 404

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
