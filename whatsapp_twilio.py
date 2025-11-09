"""
WhatsApp Bot Integration using Twilio WhatsApp API
Production-ready solution - Anyone can chat on your WhatsApp number and get bot replies
"""
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration from .env
API_URL = os.getenv("API_URL", "http://localhost:8000")
WHATSAPP_PORT = int(os.getenv("WHATSAPP_PORT", 5000))

@app.route('/whatsapp/webhook', methods=['POST'])
def whatsapp_webhook():
    """
    Handle incoming WhatsApp messages from Twilio
    This endpoint receives messages from anyone who chats on your WhatsApp number
    """
    try:
        # Get message details from Twilio
        incoming_message = request.form.get('Body', '').strip()
        from_number = request.form.get('From', '')
        to_number = request.form.get('To', '')
        
        logger.info(f"[WhatsApp] Received message from {from_number}")
        logger.info(f"[WhatsApp] Message: {incoming_message}")
        
        # Skip empty messages
        if not incoming_message:
            return _send_response("Please send a message.")
        
        # Use phone number as conversation_id for context (remove whatsapp: prefix and +)
        conversation_id = from_number.replace('whatsapp:', '').replace('+', '')
        
        # Call chatbot API
        try:
            logger.info(f"[WhatsApp] Calling API for conversation {conversation_id}")
            api_response = requests.post(
                f"{API_URL}/chat",
                json={
                    "query": incoming_message,
                    "conversation_id": conversation_id
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if api_response.status_code == 200:
                data = api_response.json()
                bot_response = data.get("response", "Sorry, I couldn't process that.")
                logger.info(f"[WhatsApp] Bot response length: {len(bot_response)} chars")
                logger.info(f"[WhatsApp] Response preview: {bot_response[:200]}...")
                logger.info(f"[WhatsApp] ✓ Response sent to {from_number}")
            else:
                logger.error(f"[WhatsApp] API Error: {api_response.status_code} - {api_response.text}")
                bot_response = "Sorry, there was an error processing your message."
                
        except requests.exceptions.Timeout:
            logger.error("API request timeout")
            bot_response = "Sorry, the request took too long. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {str(e)}")
            bot_response = "Sorry, the chatbot service is temporarily unavailable."
        
        # Send response back via Twilio
        return _send_response(bot_response)
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}", exc_info=True)
        return _send_response("Sorry, I encountered an error. Please try again later.")

def _send_response(message: str) -> Response:
    """Create Twilio response"""
    resp = MessagingResponse()
    resp.message(message)
    return Response(str(resp), mimetype='text/xml')

@app.route('/whatsapp/status', methods=['POST'])
def status_callback():
    """Handle message status updates from Twilio"""
    status = request.form.get('MessageStatus', '')
    message_sid = request.form.get('MessageSid', '')
    logger.info(f"Message status: {status} for SID: {message_sid}")
    return Response('', status=200)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "WhatsApp Twilio Bot",
        "api_url": API_URL
    }

if __name__ == '__main__':
    logger.info(f"Starting WhatsApp Bot on port {WHATSAPP_PORT}")
    logger.info(f"Chatbot API URL: {API_URL}")
    app.run(host='0.0.0.0', port=WHATSAPP_PORT, debug=False)

