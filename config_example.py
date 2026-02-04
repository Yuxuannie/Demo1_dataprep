"""
LLM Configuration for AIQC Agent
Copy this file to config.py and update with your Qwen API details
"""

# Qwen LLM Configuration (Optional)
# If not configured, agent will use template-based reasoning
QWEN_CONFIG = {
    'enabled': False,  # Set to True to enable LLM
    'api_url': 'http://your-qwen-server:8000/v1/chat/completions',
    'api_key': 'your-api-key-here',
    'model': 'qwen-72b',
    'timeout': 30,
    'temperature': 0.3,
    'max_tokens': 1000
}

# Example configurations for different deployments:

# Local Qwen deployment
# QWEN_CONFIG = {
#     'enabled': True,
#     'api_url': 'http://localhost:8000/v1/chat/completions',
#     'api_key': 'local-key',
#     'model': 'qwen-72b'
# }

# TSMC internal Qwen server (example)
# QWEN_CONFIG = {
#     'enabled': True,
#     'api_url': 'http://qwen-inference-server.tsmc.com:8080/v1/chat/completions',
#     'api_key': 'your-tsmc-api-key',
#     'model': 'qwen-72b-instruct'
# }

# Cloud Qwen service (example)
# QWEN_CONFIG = {
#     'enabled': True,
#     'api_url': 'https://api.qwen.cloud/v1/chat/completions',
#     'api_key': 'sk-xxxxxxxxxxxxxxxxxxxxxxxx',
#     'model': 'qwen-72b'
# }

# Usage in code:
# from config import QWEN_CONFIG
# 
# if QWEN_CONFIG['enabled']:
#     agent = DataPrepAgent(
#         llm_endpoint=QWEN_CONFIG['api_url'],
#         llm_api_key=QWEN_CONFIG['api_key']
#     )
# else:
#     agent = DataPrepAgent()  # Uses template-based reasoning
