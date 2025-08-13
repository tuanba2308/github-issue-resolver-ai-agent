# GitHub Issue Resolver AI Agent - Enhanced Deployment Action Plan

## Deployment Architecture Overview

```
GitHub Issue
    ↓ (Webhook + Signature Verification)
FastAPI Webhook Handler
    ↓ Publishes event to Redis Streams  ← [Redis OSS]
         ↘ Celery workers for quick, stateless jobs (labeling, comments) ← [Celery + Redis OSS]
         ↘ LangGraph agent flow for multi-step LLM logic ← [LangGraph OSS]
              ↳ Durable long-running steps wrapped in Temporal workflows ← [Temporal OSS]
                    ↕
             Cipher Memory Layer (MCP server, self-hosted) ← [Cipher OSS]
                    ↕
             Anthropic API (Claude Sonnet)
    ↑ OpenTelemetry Tracing
```

## Prerequisites and Free Tools

### Required Tools
1. **FastAPI**: Web framework for webhook handling
2. **Redis OSS**: Message queue and caching
3. **Celery**: Distributed task queue
4. **LangGraph OSS**: Agent workflow orchestration
5. **Temporal OSS**: Workflow durability and management
6. **Cipher OSS**: Memory and context layer
7. **Anthropic API** (Claude Sonnet): Language model
8. **OpenTelemetry**: Distributed tracing
9. **Prometheus**: Metrics collection
10. **Slowapi**: Rate limiting

### System Requirements
- Python 3.11+
- Docker (recommended for containerization)
- GitHub Personal Access Token
- Anthropic API Key
- Minimum 4GB RAM, 2 CPU cores

## Deployment Steps

### 1. Enhanced Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install comprehensive dependencies
pip install \
    fastapi uvicorn redis celery \
    langchain langgraph temporalio \
    anthropic github pygithub \
    pydantic prometheus_client \
    opentelemetry-api opentelemetry-sdk \
    slowapi httpx \
    pytest pytest-asyncio \
    httpx-oauth
```

### 2. Advanced Configuration Management
Create `config.py`:
```python
from pydantic import BaseSettings, SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: SecretStr
    GITHUB_TOKEN: SecretStr
    
    # Service Configurations
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    TEMPORAL_HOST: str = 'localhost'
    TEMPORAL_PORT: int = 7233
    
    # Security Settings
    WEBHOOK_SECRET: SecretStr
    JWT_SECRET: SecretStr
    
    # Cipher Memory Configuration
    CIPHER_MCP_URL: str = 'http://localhost:3000'
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        secrets_dir = '/run/secrets'

@lru_cache()
def get_settings():
    return Settings()
```

### 3. Webhook Handler with Enhanced Security
`main.py`:
```python
import json
import hmac
import hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from redis import Redis
from celery import Celery
from temporalio.client import Client as TemporalClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configuration
from config import get_settings

# Setup OpenTelemetry Tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))
tracer = trace.get_tracer(__name__)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware for host validation
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', 'your-domain.com'])

# Redis and Celery Initialization
redis_client = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
celery_app = Celery('tasks', broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}')

@app.post('/webhook/github')
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW}")
async def github_webhook(request: Request):
    with tracer.start_as_current_span("github_webhook"):
        # Verify GitHub webhook signature
        payload_body = await request.body()
        signature_header = request.headers.get('X-Hub-Signature-256')
        
        if not signature_header:
            raise HTTPException(status_code=403, detail="No signature found")
        
        expected_signature = 'sha256=' + hmac.new(
            settings.WEBHOOK_SECRET.get_secret_value().encode(),
            payload_body,
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, signature_header):
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        try:
            payload = json.loads(payload_body)
            # Publish to Redis Stream with error handling
            redis_client.xadd('github_events', {
                'payload': json.dumps(payload),
                'timestamp': int(time.time())
            })
            
            # Trigger initial processing tasks
            celery_app.send_task('tasks.process_github_event', args=[payload])
            
            return {"status": "accepted", "message": "Event processed successfully"}
        
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        except Exception as e:
            # Log the error for debugging
            logger.error(f"Webhook processing error: {e}")
            raise HTTPException(status_code=500, detail="Internal processing error")
```

### 4. Advanced Celery Worker with Retry
`tasks.py`:
```python
from celery import Celery
from celery.exceptions import MaxRetriesExceededError
from temporalio.workflow import workflow_method
import time

celery_app = Celery('tasks', broker='redis://localhost:6379')
celery_app.conf.update(
    task_acks_late=True,
    task_retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.6,
    }
)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def process_github_event(self, payload):
    try:
        # Process event with robust error handling
        # Example: auto-labeling, initial classification
        pass
    except Exception as exc:
        try:
            self.retry(exc=exc)
        except MaxRetriesExceededError:
            # Log failure, potentially notify admin
            logger.error(f"Failed to process event after retries: {payload}")
```

### 5. Enhanced Cipher Memory Layer
```python
from typing import Dict, List, Optional
import time
import hashlib

class AdvancedCipherMemory:
    def __init__(self, url: str):
        self.url = url
        self.client = httpx.AsyncClient()
    
    async def store_context(self, context: Dict, tags: List[str] = None):
        """Store context with advanced metadata"""
        memory = {
            'id': self._generate_id(context),
            'content': context,
            'metadata': {
                'created_at': time.time(),
                'tags': tags or [],
                'source': 'github_webhook'
            }
        }
        await self._send_to_mcp(memory)
    
    async def retrieve_similar_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve similar context using semantic search"""
        payload = {
            'query': query,
            'top_k': top_k,
            'method': 'semantic_search'
        }
        response = await self.client.post(f"{self.url}/search", json=payload)
        return response.json().get('results', [])
    
    def _generate_id(self, context: Dict) -> str:
        """Generate unique ID for context"""
        content_hash = hashlib.md5(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        return f"context_{content_hash}"
```

### 6. Docker Compose for Full Environment
`docker-compose.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  temporal:
    image: temporalio/auto-setup:latest
    ports:
      - "7233:7233"
    environment:
      - DYNAMIC_CONFIG_FILE_PATH=config/dynamicconfig/development.yaml
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "6831:6831/udp"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    restart: unless-stopped

  agent-cluster:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
    depends_on:
      - redis
      - temporal
      - jaeger
    restart: unless-stopped

volumes:
  redis_data:
```

### 7. Comprehensive Testing Strategy
`tests/test_webhook.py`:
```python
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_github_webhook_signature():
    """Test webhook signature verification"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test invalid signature
        response = await ac.post(
            "/webhook/github", 
            json={"test": "payload"},
            headers={"X-Hub-Signature-256": "invalid_signature"}
        )
        assert response.status_code == 403

@pytest.mark.asyncio
async def test_rate_limiting():
    """Verify rate limiting functionality"""
    # Implement rate limit test
    pass
```

## Enhanced Monitoring and Observability
- OpenTelemetry for distributed tracing
- Prometheus metrics collection
- Jaeger for trace visualization
- Comprehensive logging with structured log formats

## Security Enhancements
- GitHub webhook signature verification
- Rate limiting with `slowapi`
- Environment-based configuration management
- Secrets management using `.env` and `/run/secrets`

## Performance Optimization
- Async programming model
- Efficient Redis Streams usage
- Celery task retry mechanisms
- Distributed tracing for performance insights

## Deployment Considerations
- Containerized microservices architecture
- Scalable design with independent services
- Easy horizontal scaling capabilities

## Estimated Deployment Time
- Initial setup: 4-6 hours
- Configuration and testing: 8-12 hours
- Performance tuning: 4-6 hours

## Future Roadmap
1. Implement advanced caching strategies
2. Add machine learning-based anomaly detection
3. Develop more sophisticated agent workflows
4. Enhance multi-repository support
5. Implement advanced security scanning

## Cost and Resource Management
- Fully open-source infrastructure
- Pay only for Anthropic API usage
- Minimal cloud resource requirements
- Horizontal scaling potential

## Continuous Improvement
- Regular dependency updates
- Security patch monitoring
- Performance benchmarking
- Community feedback integration

## Rollback and Recovery
- Keep configuration in version control
- Maintain regular backups of Cipher memory layer
- Use Temporal for workflow resumability