"""
START FASTAPI SERVER
====================
Start the recommendation API server

Usage:
    python scripts/online/start_server.py
    
    # With custom config
    python scripts/online/start_server.py --config configs/config_online.yaml
    
    # Development mode with reload
    python scripts/online/start_server.py --reload
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import logging

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    print("❌ uvicorn not available. Install: pip install uvicorn")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Start Recommendation API Server')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_online.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Host to bind (overrides config)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to bind (overrides config)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of workers (overrides config)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload (development)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level'
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Get server config
    api_config = config.get('api', {})
    
    # Server parameters (CLI args override config)
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or api_config.get('port', 8010)
    workers = args.workers or api_config.get('workers', 4)
    reload = args.reload or api_config.get('reload', False)
    
    # Display banner
    print("\n" + "="*70)
    print("🚀 RECOMMENDATION API SERVER")
    print("="*70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers if not reload else 1} {'(auto-reload enabled)' if reload else ''}")
    print(f"Log level: {args.log_level}")
    print("="*70 + "\n")
    
    # Start server
    uvicorn.run(
        "scripts.online.api:app",
        host=host,
        port=port,
        workers=1 if reload else workers,  # Must be 1 for reload
        reload=reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()