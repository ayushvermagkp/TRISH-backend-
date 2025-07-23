from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import requests
import logging
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
MODEL_NAME = "shisa-ai/shisa-v2-llama3.3-70b:free"
API_KEY = os.getenv("OPENROUTER_API_KEY_PRIMARY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "https://your-frontend-site.onrender.com")
YOUR_SITE_NAME = "TRISH Discussion Platform"

def call_openrouter_api(messages, system_prompt=None):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME
    }

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": MODEL_NAME,
        "messages": full_messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logging.error(f"API failed: {e}")
        return None

@app.route('/api/chat', methods=['POST'])
@limiter.limit("15/minute")
def chat():
    data = request.json
    messages = data.get('messages', [])
    discussion_topic = data.get('discussion_topic', 'General discussion')

    system_prompt = f"""You are TRISH, an AI discussion facilitator. Current topic: {discussion_topic}
Your role:
- Ask probing questions to deepen understanding
- Identify common ground between participants
- Challenge assumptions constructively
- Ensure balanced participation
- Keep discussion focused and productive

Respond naturally using markdown when helpful. Be concise and engaging."""

    response = call_openrouter_api(messages, system_prompt)

    if response:
        return jsonify({"response": response})
    else:
        return jsonify({"error": "All AI services are currently unavailable"}), 503

@app.route('/api/generate-conclusion', methods=['POST'])
@limiter.limit("10/minute")
def generate_conclusion():
    data = request.json
    messages = data.get('messages', [])
    discussion_topic = data.get('discussion_topic', 'General discussion')

    system_prompt = f"""Generate comprehensive conclusion for: {discussion_topic}
Required format:

## Key Points Summary
- 3-5 main discussion outcomes
- Focus on concrete results

## Major Insights
- Notable observations
- Unexpected findings
- Participant breakthroughs

## Action Items
- Clear next steps with owners
- Specific deadlines
- Success metrics

Use markdown formatting and maintain professional tone."""

    response = call_openrouter_api(messages, system_prompt)

    if response:
        return jsonify({
            "conclusion": response,
            "structured": parse_conclusion(response)
        })
    else:
        return jsonify({"error": "All conclusion services are unavailable"}), 503

def parse_conclusion(text):
    sections = {}
    current_section = None

    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            current_section = line[3:].strip()
            sections[current_section] = []
        elif current_section and line.startswith('- '):
            sections[current_section].append(line[2:].strip())

    return sections

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "site": YOUR_SITE_NAME,
        "key_status": "available" if API_KEY else "missing"
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
