import os
import asyncio
import aiohttp
from urllib.robotparser import RobotFileParser
from collections import deque
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import io
from PIL import Image  # pip install pillow
import hashlib
import re

load_dotenv()

# Config
INDEX_NAME = "cc"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PAGES = 50  # Scalability limit
MAX_DEPTH = 2
BATCH_SIZE = 100  # For Pinecone upserts
USE_VISION = True  # Set False to skip GPT-4V (saves cost)

def get_base_url(url: str) -> str:
    from urllib.parse import urlparse
    return f"{urlparse(url).scheme}://{urlparse(url).netloc}"

def generate_namespace(url: str) -> str:
    """Generate a unique namespace from a URL using hash of base URL."""
    base_url = get_base_url(url)
    # Create a hash of the base URL for namespace
    # Namespace must be alphanumeric and underscores, max 40 chars
    namespace_hash = hashlib.md5(base_url.encode()).hexdigest()[:16]
    # Sanitize base URL for readable namespace
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', base_url.replace('https://', '').replace('http://', '').replace('www.', ''))
    sanitized = sanitized[:20]  # Limit length
    namespace = f"{sanitized}_{namespace_hash}"
    return namespace

async def can_crawl(url: str, base_url: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(f"{base_url}/robots.txt")
    try:
        await rp.read()  # Async? Use sync for simplicity
    except:
        pass
    return rp.can_fetch("*", url)

def extract_links(soup: BeautifulSoup, base_url: str, current_url: str) -> List[str]:
    from urllib.parse import urljoin, urlparse
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        # Handle relative URLs
        if href.startswith('http'):
            absolute_url = href
        elif href.startswith('/'):
            absolute_url = urljoin(base_url, href)
        else:
            absolute_url = urljoin(current_url, href)
        
        # Only add links from the same domain
        if get_base_url(absolute_url) == get_base_url(current_url):
            links.add(absolute_url)
    return list(links)[:10]  # Limit per page

async def describe_image(image_url: str, alt: str) -> str:
    if not USE_VISION or alt != "No alt text":  # Skip if good alt
        return alt
    try:
        # Fetch image bytes
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    img_bytes = await resp.read()
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.size[0] < 100 or img.size[1] < 100:  # Skip icons
                        return "Decorative image"
        
        # Use GPT-4V for description
        llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=50)
        response = llm.invoke([
            {"type": "text", "text": "Describe this image briefly for searchability."},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
        return response.content
    except Exception:
        return alt

async def scrape_page(page, url: str) -> tuple[str, List[Dict[str, str]]]:
    content = await page.content()
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract text
    text = soup.get_text(separator=' ', strip=True)
    
    # Extract and describe images
    images = []
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            if not src.startswith('http'):
                src = url + src if src.startswith('/') else url.rsplit('/', 1)[0] + '/' + src
            alt = img.get('alt', 'No alt text')
            desc = await describe_image(src, alt)
            images.append({'url': src, 'description': desc})
    
    return text, images

async def crawl_and_process(start_url: str):
    base_url = get_base_url(start_url)
    # Generate unique namespace for this URL
    namespace = generate_namespace(start_url)
    print(f"Using namespace: {namespace} for URL: {start_url}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=INDEX_NAME, dimension=384, metric='cosine')
    # Create vectorstore with namespace
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings,
        namespace=namespace
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    visited = set()
    queue = deque([(start_url, 0)])  # (url, depth)
    all_chunks = []
    all_metadata = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (compatible; Bot)")  # Polite UA
        page = await context.new_page()
        
        while queue and len(visited) < MAX_PAGES:
            url, depth = queue.popleft()
            if url in visited or depth > MAX_DEPTH:
                continue
            visited.add(url)
            
            if not await can_crawl(url, base_url):
                print(f"Skipped {url} (robots.txt)")
                continue
            
            print(f"Scraping {url} (depth {depth}, total {len(visited)}/{MAX_PAGES})")
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                text, images = await scrape_page(page, url)
                
                # Chunk text
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    chunk += f" [Page: {url}]"  # Context
                
                # Add image descriptions as chunks
                for img in images:
                    img_chunk = f"Image on {url}: {img['description']}"
                    chunks.append(img_chunk)
                
                # Collect for batching
                metadata = [{'url': url, 'type': 'text' if 'Page:' in c else 'image'} for c in chunks]
                all_chunks.extend(chunks)
                all_metadata.extend(metadata)
                
                # Enqueue links
                if depth < MAX_DEPTH:
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    found_links = extract_links(soup, base_url, url)
                    print(f"  Found {len(found_links)} links on {url}")
                    new_links_count = 0
                    for link in found_links:
                        if link not in visited:
                            queue.append((link, depth + 1))
                            new_links_count += 1
                    print(f"  Added {new_links_count} new links to queue (queue size: {len(queue)})")
                
                # Rate limit
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error on {url}: {e}")
        
        await browser.close()
    
    # Batch upsert to Pinecone in the namespace
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_texts = all_chunks[i:i+BATCH_SIZE]
        batch_meta = all_metadata[i:i+BATCH_SIZE]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_meta)
        print(f"Upserted batch {i//BATCH_SIZE + 1} ({len(batch_texts)} items) to namespace '{namespace}'")
    
    print(f"Done! Stored {len(all_chunks)} items from {len(visited)} pages in namespace '{namespace}'.")

# Run
async def main():
    start_url = input("Enter starting URL: ")
    await crawl_and_process(start_url)

if __name__ == "__main__":
    asyncio.run(main())