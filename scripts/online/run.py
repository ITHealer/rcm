"""
SIMPLE SERVER RUNNER
====================
Easy way to run the API server

Usage:
    python run.py
    
    # With custom port
    python run.py --port 8080
    
    # Development mode (auto-reload)
    python run.py --dev
"""

import os
import sys
from pathlib import Path
import argparse

# Add project root
sys.path.append(str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def main():
    """Run the server"""
    parser = argparse.ArgumentParser(description='Run Recommendation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8010, help='Port')
    parser.add_argument('--dev', action='store_true', help='Development mode (auto-reload)')
    parser.add_argument('--no-redis', action='store_true', help='Disable Redis')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['API_HOST'] = args.host
    os.environ['API_PORT'] = str(args.port)
    
    if args.no_redis:
        os.environ.pop('REDIS_HOST', None)
    
    # Print banner
    print("\n" + "="*70)
    print("🚀 RECOMMENDATION API SERVER")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Mode: {'Development (auto-reload)' if args.dev else 'Production'}")
    print(f"Redis: {'disabled' if args.no_redis else 'enabled'}")
    print("="*70 + "\n")
    
    # Import and run
    try:
        import uvicorn
        
        uvicorn.run(
            "scripts.online.api:app",
            host=args.host,
            port=args.port,
            reload=args.dev,
            log_level="info"
        )
        
    except ImportError:
        print("❌ Error: uvicorn not installed")
        print("Install: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()