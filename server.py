#!/usr/bin/env python3
"""
Launcher script for the playwright MCP server
This file allows the deployment system to find and run the server
"""

import sys
import asyncio
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the actual server
from playwright_server.server import main

if __name__ == "__main__":
    # Check for PORT environment variable to determine transport
    port = os.environ.get('PORT')
    
    if port:
        # If PORT is set, use SSE transport (more compatible than HTTP)
        sys.argv.extend(['--transport', 'sse', '--port', port, '--host', '0.0.0.0'])
        print(f"🌐 Starting SSE server on port {port}")
    else:
        # Default to stdio transport
        print("📡 Starting stdio server")
    
    asyncio.run(main()) 