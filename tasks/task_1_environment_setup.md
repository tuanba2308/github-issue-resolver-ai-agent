# Task 1: Enhanced Environment Setup for GitHub Issue Resolver AI Agent

## Objective
Create a reproducible, secure, and comprehensive development environment for the GitHub Issue Resolver AI Agent project.

## Prerequisites
- Python 3.11+
- pip
- virtualenv or venv
- git
- Docker (optional, but recommended)

## Steps

### 1. Python Virtual Environment
```bash
# Create virtual environment
python3.11 -m venv ai_agent_env

# Activate the environment
source ai_agent_env/bin/activate
```

### 2. Install Core Dependencies
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install comprehensive dependencies
pip install \
    fastapi uvicorn redis celery \
    langchain langgraph temporalio \
    anthropic github pygithub \
    pydantic prometheus_client \
    opentelemetry-api opentelemetry-sdk \
    slowapi httpx \
    pytest pytest-asyncio \
    httpx-oauth \
    python-dotenv
```

### 3. Create Project Structure
```bash
# Create project directory
mkdir -p github-issue-resolver/src
mkdir -p github-issue-resolver/tests
mkdir -p github-issue-resolver/config

# Initialize git repository
git init
```

### 4. Environment Configuration
```bash
# Create .env file template
cat > github-issue-resolver/.env.template << EOL
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key
GITHUB_TOKEN=your_github_token

# Service Configurations
REDIS_HOST=localhost
REDIS_PORT=6379
TEMPORAL_HOST=localhost
TEMPORAL_PORT=7233

# Security Settings
WEBHOOK_SECRET=your_github_webhook_secret
JWT_SECRET=your_jwt_secret

# Cipher Memory Configuration
CIPHER_MCP_URL=http://localhost:3000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
EOL

# Create .gitignore
cat > github-issue-resolver/.gitignore << EOL
# Virtual environment
ai_agent_env/
venv/

# Environment files
.env
*.env

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Logs
*.log

# IDE settings
.vscode/
.idea/

# Temporary files
*.swp
*.swo
EOL
```

### 5. Create Basic Configuration File
```python
# github-issue-resolver/config/settings.py
from pydantic import BaseSettings, SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    ANTHROPIC_API_KEY: SecretStr
    GITHUB_TOKEN: SecretStr
    
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    
    WEBHOOK_SECRET: SecretStr
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

@lru_cache()
def get_settings():
    return Settings()
```

### 6. Initial Development Validation
```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify pip installations
pip list

# Create a simple test to ensure configuration works
python -c "from config.settings import get_settings; print(get_settings())"
```

## Validation Checklist
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] Project structure set up
- [ ] .env template created
- [ ] .gitignore configured
- [ ] Basic configuration file created
- [ ] Initial environment validation complete

## Potential Challenges and Mitigations
1. Python version compatibility
   - Ensure Python 3.11 is installed
   - Use pyenv or conda for version management if needed

2. Dependency conflicts
   - Use `pip freeze > requirements.txt` to lock dependency versions
   - Consider using Poetry or Pipenv for more robust dependency management

3. Secret management
   - Never commit .env files with real secrets
   - Use secret management tools in production

## Next Steps
- Set up Docker configuration
- Implement initial webhook handler
- Configure initial testing framework

## Estimated Time
- Initial setup: 1-2 hours
- Troubleshooting: 30 minutes - 1 hour

## Notes
Ensure all team members follow this setup process to maintain consistent development environments.