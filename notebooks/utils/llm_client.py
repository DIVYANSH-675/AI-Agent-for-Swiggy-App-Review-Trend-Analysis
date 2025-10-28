"""
Unified LLM Client for OpenAI and Ollama
Provides caching, retries, and batch processing
"""
import json
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import time
from threading import Lock

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env if available
except ImportError:
    pass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from tqdm import tqdm


class LLMClient:
    """Megallm + Ollama client with caching"""
    
    def __init__(self, provider='megallm', model='gpt-3.5-turbo', cache_db='cache.db'):
        """
        Initialize LLM client
        
        Args:
            provider: 'megallm' or 'ollama'
            model: Model name (e.g., 'gpt-3.5-turbo', 'llama3.1:8b')
            cache_db: Path to SQLite cache database
        """
        self.provider = provider
        self.model = model
        self.cache_db = cache_db
        
        # Setup caching
        self.cache = sqlite3.connect(cache_db, check_same_thread=False)
        self.cache_lock = Lock()
        self.cache.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL
            )
        """)
        self.cache.commit()
        
        # Initialize provider
        if provider == 'megallm':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")

            api_key = os.getenv('MEGALLM_API_KEY', 'sk-mega-5c718f7e9327ca90c8dbf159e39a4192407c48474a922ec5fa6a91027a247d1a')
            base_url = os.getenv('MEGALLM_BASE_URL', 'https://ai.megallm.io/v1')
            organization = os.getenv('MEGALLM_ORG')
            client_kwargs = {'api_key': api_key, 'base_url': base_url}
            if organization:
                client_kwargs['organization'] = organization
            self.client = OpenAI(**client_kwargs)

        elif provider == 'ollama':
            if not HTTPX_AVAILABLE:
                raise ImportError("httpx package not installed for Ollama")
            self.client = httpx.Client(base_url='http://localhost:11434', timeout=300.0)

        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        print(f"✓ Initialized {provider} client with model {model}")
    
    def _hash_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Generate hash for caching"""
        content = f"{system_prompt}|{user_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Retrieve from cache"""
        with self.cache_lock:
            cursor = self.cache.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
        return row[0] if row else None
    
    def _save_to_cache(self, key: str, value: str):
        """Save to cache"""
        with self.cache_lock:
            self.cache.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                (key, value, time.time())
            )
            self.cache.commit()
    
    def complete(self, system_prompt: str, user_prompt: str, 
                 temperature=0.3, response_format='json', use_cache=True):
        """
        Complete a prompt with caching
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            response_format: 'json' or 'text'
            use_cache: Whether to use cache
            
        Returns:
            Parsed JSON or raw text
        """
        # Check cache
        if use_cache:
            key = self._hash_prompt(system_prompt, user_prompt)
            cached = self._get_from_cache(key)
            if cached:
                if response_format == 'json':
                    return json.loads(cached)
                return cached
        
        # Call LLM
        if self.provider in ('megallm', 'openai'):
            response = self._call_openai(system_prompt, user_prompt, temperature, response_format)
        else:
            response = self._call_ollama(system_prompt, user_prompt, temperature, response_format)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(key, json.dumps(response) if isinstance(response, dict) else response)
        
        return response
    
    def _call_openai(self, system_prompt: str, user_prompt: str, 
                    temperature: float, response_format: str):
        """Call OpenAI-compatible chat completion API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        extra_kwargs = {}
        if response_format == 'json':
            extra_kwargs['response_format'] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            **extra_kwargs
        )
        
        content = response.choices[0].message.content
        
        if response_format == 'json':
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON: {content[:200]}")
                return {"error": content}
        
        return content
    
    def _call_ollama(self, system_prompt: str, user_prompt: str, 
                    temperature: float, response_format: str):
        """Call Ollama API"""
        prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        response = self.client.post(
            f'/api/generate',
            json={
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': temperature}
            }
        ).json()
        
        content = response['response']
        
        if response_format == 'json':
            try:
                # Try to extract JSON from response
                content = content.strip()
                if content.startswith('```'):
                    # Remove markdown code blocks
                    content = '\n'.join(content.split('\n')[1:-1])
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON: {content[:200]}")
                return {"error": content}
        
        return content
    
    def batch_complete(self, prompts: List[Dict], batch_size=10):
        """
        Process multiple prompts with progress bar
        
        Args:
            prompts: List of {system_prompt, user_prompt} dicts
            batch_size: Batch size for progress updates
            
        Returns:
            List of results
        """
        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
            batch = prompts[i:i+batch_size]
            batch_results = []
            for item in batch:
                try:
                    result = self.complete(
                        item['system_prompt'],
                        item['user_prompt'],
                        item.get('temperature', 0.3),
                        item.get('response_format', 'json')
                    )
                    batch_results.append(result)
                    time.sleep(0.05)  # Rate limiting
                except Exception as e:
                    print(f"\nError processing prompt: {e}")
                    batch_results.append(None)
            results.extend(batch_results)
        
        return results
    
    def clear_cache(self):
        """Clear all cache entries"""
        with self.cache_lock:
            self.cache.execute("DELETE FROM cache")
            self.cache.commit()
        print("Cache cleared")
    
    def cache_stats(self):
        """Get cache statistics"""
        with self.cache_lock:
            cursor = self.cache.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
        return {"cached_entries": count}


if __name__ == "__main__":
    # Test the client
    client = LLMClient(provider='openai', model='gpt-3.5-turbo')
    
    test_result = client.complete(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, World!'",
        response_format='text'
    )
    
    print(f"Test result: {test_result}")

