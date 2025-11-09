"""
WhatsApp Bot using Personal Number (WhatsApp Web.js)
⚠️ LIMITATIONS:
- Uses your personal WhatsApp number
- Requires your phone to be connected to internet
- Unofficial method (may get banned with heavy use)
- Only works when bot is running
- Not suitable for high-volume production use
"""
import subprocess
import sys
import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
NODE_SCRIPT = "whatsapp_personal_bot.js"

def create_node_script():
    """Create Node.js script for WhatsApp Web"""
    script_content = """const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const http = require('http');
const https = require('https');
const { URL } = require('url');

const API_URL = process.env.API_URL || 'http://localhost:8000';

const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: './whatsapp_session'
    }),
    puppeteer: {
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }
});

client.on('qr', (qr) => {
    console.log('QR_CODE:', qr);
    qrcode.generate(qr, { small: true });
    console.log('\\nScan this QR code with WhatsApp to connect!\\n');
});

client.on('ready', () => {
    console.log('READY: WhatsApp client is ready!');
    console.log('Bot is now listening for messages...\\n');
});

client.on('message', async (message) => {
    try {
        // Skip group messages
        if (message.from.includes('@g.us')) {
            return;
        }

        const fromNumber = message.from;
        const messageText = message.body.trim();

        if (!messageText) {
            return;
        }

        console.log(`Received from ${fromNumber}: ${messageText}`);

        // Use phone number as conversation_id
        const conversationId = fromNumber.replace('@c.us', '');

        // Call chatbot API
        try {
            const apiUrl = new URL(`${API_URL}/chat`);
            const postData = JSON.stringify({
                query: messageText,
                conversation_id: conversationId
            });

            const options = {
                hostname: apiUrl.hostname,
                port: apiUrl.port || (apiUrl.protocol === 'https:' ? 443 : 80),
                path: apiUrl.pathname,
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(postData)
                }
            };

            const requestModule = apiUrl.protocol === 'https:' ? https : http;

            const response = await new Promise((resolve, reject) => {
                const req = requestModule.request(options, (res) => {
                    let data = '';
                    res.on('data', (chunk) => { data += chunk; });
                    res.on('end', () => {
                        resolve({ statusCode: res.statusCode, data: data });
                    });
                });

                req.on('error', (error) => {
                    reject(error);
                });

                req.write(postData);
                req.end();
            });

            if (response.statusCode === 200) {
                const data = JSON.parse(response.data);
                const botResponse = data.response || "Sorry, I couldn't process that.";

                // Send response back
                await client.sendMessage(fromNumber, botResponse);
                console.log(`Sent response to ${fromNumber}`);
            } else {
                console.error(`API Error: ${response.statusCode} - ${response.data}`);
                await client.sendMessage(
                    fromNumber,
                    "Sorry, there was an error processing your message."
                );
            }
        } catch (error) {
            console.error(`Error calling API: ${error.message}`);
            await client.sendMessage(
                fromNumber,
                "Sorry, the chatbot service is temporarily unavailable."
            );
        }
    } catch (error) {
        console.error(`Error handling message: ${error.message}`);
    }
});

client.on('disconnected', (reason) => {
    console.log('DISCONNECTED:', reason);
    console.log('Reconnecting...');
    client.initialize();
});

client.on('auth_failure', (msg) => {
    console.error('AUTH_FAILURE:', msg);
});

console.log('Initializing WhatsApp client...');
client.initialize();
"""
    
    with open(NODE_SCRIPT, 'w') as f:
        f.write(script_content)
    print(f"✓ Created {NODE_SCRIPT}")

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Node.js is not installed!")
        print("Please install Node.js from https://nodejs.org/")
        return False
    return False

def check_dependencies():
    """Check if Node.js dependencies are installed"""
    if not os.path.exists('node_modules'):
        return False
    return True

def install_dependencies():
    """Install Node.js dependencies"""
    print("Installing Node.js dependencies...")
    try:
        subprocess.run(['npm', 'install', 'whatsapp-web.js', 'qrcode-terminal'], check=True)
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def run_bot():
    """Run the WhatsApp bot"""
    print("\n" + "="*60)
    print("WhatsApp Bot (Personal Number) - Starting...")
    print("="*60 + "\n")
    
    # Check prerequisites
    if not check_node_installed():
        sys.exit(1)
    
    if not check_dependencies():
        if not install_dependencies():
            sys.exit(1)
    
    # Create Node.js script
    create_node_script()
    
    # Check if API is running
    api_url = os.getenv("API_URL", "http://localhost:8000")
    print(f"✓ API URL: {api_url}")
    print("⚠️  Make sure your chatbot API is running on this URL!\n")
    
    print("⚠️  IMPORTANT LIMITATIONS:")
    print("   - Uses your personal WhatsApp number")
    print("   - Requires your phone to be connected to internet")
    print("   - Unofficial method (may get banned with heavy use)")
    print("   - Only works when bot is running")
    print("   - Not suitable for high-volume production\n")
    
    # Run the Node.js bot
    print("Starting WhatsApp bot...")
    print("Scan the QR code when it appears!\n")
    
    try:
        env = os.environ.copy()
        env['API_URL'] = api_url
        subprocess.run(['node', NODE_SCRIPT], env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_bot()

