import asyncio

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

server = Server("playwright-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources.
    Each note is exposed as a resource with a custom note:// URI scheme.
    """
    return []

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content by its URI.
    The note name is extracted from the URI host component.
    """
    raise ValueError(f"Unsupported URI scheme: {uri.scheme}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return []

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    raise ValueError(f"Unknown prompt: {name}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        # types.Tool(
        #     name="playwright_new_session",
        #     description="Create a new browser session",
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "url": {"type": "string", "description": "Initial URL to navigate to"}
        #         }
        #     }
        # ),
        types.Tool(
            name="playwright_navigate",
            description="Navigate to a URL,thip op will auto create a session",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="playwright_screenshot",
            description="Take a screenshot of the current page or a specific element",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "selector": {"type": "string", "description": "CSS selector for element to screenshot,null is full page"},
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="playwright_click",
            description="Click an element on the page using CSS selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for element to click"}
                },
                "required": ["selector"]
            }
        ),
        types.Tool(
            name="playwright_fill",
            description="Fill out an input field",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for input field"},
                    "value": {"type": "string", "description": "Value to fill"}
                },
                "required": ["selector", "value"]
            }
        ),
        types.Tool(
            name="playwright_evaluate",
            description="Execute JavaScript in the browser console",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "JavaScript code to execute"}
                },
                "required": ["script"]
            }
        ),
        types.Tool(
            name="playwright_click_text",
            description="Click an element on the page by its text content",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text content of the element to click"}
                },
                "required": ["text"]
            }
        ),
         types.Tool(
            name="playwright_get_text_content",
            description="Get the text content of all elements",
            inputSchema={
                "type": "object",
                "properties": {
                },
            }
        ),
        types.Tool(
            name="playwright_get_html_content",
            description="Get the HTML content of the page",
             inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the element"}
                },
                "required": ["selector"]
            }
        )
    ]

import uuid
from playwright.async_api import async_playwright
import base64
import os

import asyncio

# Shared session storage for all tool handlers
_sessions: dict[str, any] = {}
_playwright: any = None

def update_page_after_click(func):
    async def wrapper(self, name: str, arguments: dict | None):
        if not _sessions:
            return [types.TextContent(type="text", text="No active session. Please create a new session first.")]
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        
        new_page_future = asyncio.ensure_future(page.context.wait_for_event("page", timeout=3000))
        
        result = await func(self, name, arguments)
        try:
            new_page = await new_page_future
            await new_page.wait_for_load_state()
            _sessions[session_id]["page"] = new_page
        except:
            pass
            # if page.url != _sessions[session_id]["page"].url:
            #     await page.wait_for_load_state()
            #     _sessions[session_id]["page"] = page
        
        return result
    return wrapper

async def _ensure_valid_session():
    """Ensure we have a valid browser session, create one if needed"""
    global _sessions, _playwright
    
    # Check if we have a valid session and browser
    if _sessions:
        session_id = list(_sessions.keys())[-1]
        try:
            # Test if the browser is still alive by checking if page is still connected
            page = _sessions[session_id]["page"]
            browser = _sessions[session_id]["browser"]
            
            # More reliable check - see if browser is connected
            if not page.is_closed() and browser.is_connected():
                return  # Session is valid
        except Exception as e:
            # Browser/page is closed, clear sessions
            print(f"Browser session validation failed: {e}, creating new session...")
        
        # If we get here, session is invalid - clear it
        _sessions.clear()
        if _playwright:
            try:
                await _playwright.stop()
            except:
                pass
            _playwright = None
    
    if not _sessions:
        # Create new session
        print("Creating new browser session...")
        _playwright = await async_playwright().start()
        browser = await _playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {"browser": browser, "page": page}

class ToolHandler:
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        raise NotImplementedError

class NewSessionToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        global _sessions, _playwright
        
        _playwright = await async_playwright().start()
        browser = await _playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {"browser": browser, "page": page}
        url = arguments.get("url")
        if url:
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            await page.goto(url)
        return [types.TextContent(type="text", text="succ")]

class NavigateToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        url = arguments.get("url")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        await page.goto(url)
        text_content = await GetTextContentToolHandler().handle("", {})
        return [types.TextContent(type="text", text=f"Navigated to {url}\npage_text_content[:200]:\n\n{text_content[0].text[:200]}")]

class ScreenshotToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        name = arguments.get("name")
        selector = arguments.get("selector")
        # full_page = arguments.get("fullPage", False)
        if selector:
            element = await page.locator(selector)
            await element.screenshot(path=f"{name}.png")
        else:
            await page.screenshot(path=f"{name}.png", full_page=True)
        with open(f"{name}.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove(f"{name}.png")
        return [types.ImageContent(type="image", data=encoded_string, mimeType="image/png")]

class ClickToolHandler(ToolHandler):
    @update_page_after_click
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        selector = arguments.get("selector")
        await page.locator(selector).click()
        return [types.TextContent(type="text", text=f"Clicked element with selector {selector}")]

class FillToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        selector = arguments.get("selector")
        value = arguments.get("value")
        await page.locator(selector).fill(value)
        return [types.TextContent(type="text", text=f"Filled element with selector {selector} with value {value}")]

class EvaluateToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        script = arguments.get("script")
        result = await page.evaluate(script)
        return [types.TextContent(type="text", text=f"Evaluated script, result: {result}")]

class ClickTextToolHandler(ToolHandler):
    @update_page_after_click
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        text = arguments.get("text")
        await page.locator(f"text={text}").nth(0).click()
        return [types.TextContent(type="text", text=f"Clicked element with text {text}")]

class GetTextContentToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        # text_contents = await page.locator('body').all_inner_texts()


        async def get_unique_texts_js(page):
            unique_texts = await page.evaluate('''() => {
            var elements = Array.from(document.querySelectorAll('*')); // 先选择所有元素，再进行过滤
            var uniqueTexts = new Set();

            for (var element of elements) {
                if (element.offsetWidth > 0 || element.offsetHeight > 0) { // 判断是否可见
                    var childrenCount = element.querySelectorAll('*').length;
                    if (childrenCount <= 3) {
                        var innerText = element.innerText ? element.innerText.trim() : '';
                        if (innerText && innerText.length <= 1000) {
                            uniqueTexts.add(innerText);
                        }
                        var value = element.getAttribute('value');
                        if (value) {
                            uniqueTexts.add(value);
                        }
                    }
                }
            }
            //console.log( Array.from(uniqueTexts));
            return Array.from(uniqueTexts);
        }
        ''')
            return unique_texts

        # 使用示例
        text_contents = await get_unique_texts_js(page)



        return [types.TextContent(type="text", text=f"Text content of all elements: {text_contents}")]

class GetHtmlContentToolHandler(ToolHandler):
    async def handle(self, name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        await _ensure_valid_session()
        session_id = list(_sessions.keys())[-1]
        page = _sessions[session_id]["page"]
        selector = arguments.get("selector")
        html_content = await page.locator(selector).inner_html()
        return [types.TextContent(type="text", text=f"HTML content of element with selector {selector}: {html_content}")]


tool_handlers = {
    "playwright_navigate": NavigateToolHandler(),
    "playwright_screenshot": ScreenshotToolHandler(),
    "playwright_click": ClickToolHandler(),
    "playwright_fill": FillToolHandler(),
    "playwright_evaluate": EvaluateToolHandler(),
    "playwright_click_text": ClickTextToolHandler(),
    "playwright_get_text_content": GetTextContentToolHandler(),
    "playwright_get_html_content": GetHtmlContentToolHandler(),
    "playwright_new_session":NewSessionToolHandler(),
}


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name in tool_handlers:
        return await tool_handlers[name].handle(name, arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Playwright MCP Server')
    parser.add_argument('--transport', choices=['stdio', 'http', 'sse'], default='stdio',
                       help='Transport type: stdio (default), http, or sse')
    parser.add_argument('--host', default='localhost', 
                       help='Host for HTTP transport (default: localhost)')
    parser.add_argument('--port', type=int, default=3000,
                       help='Port for HTTP transport (default: 3000)')
    
    args = parser.parse_args()
    
    if args.transport in ['http', 'sse']:
        # HTTP/SSE transport using FastAPI
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import json
        import asyncio
        
        app = FastAPI(title="Playwright MCP Server", version="0.1.0")
        
        # Add CORS middleware for OpenAI compatibility
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, be more specific
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        if args.transport == 'sse':
            @app.post("/sse")
            async def sse_endpoint(request: Request):
                """Handle MCP requests over Server-Sent Events (OpenAI format)"""
                try:
                    data = await request.json()
                    
                    # Handle different MCP method types
                    method = data.get("method")
                    params = data.get("params", {})
                    request_id = data.get("id")
                    
                    async def generate_response():
                        if method == "tools/list":
                            # Get available tools
                            tools = await handle_list_tools()
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {"tools": [tool.model_dump() for tool in tools]}
                            }
                            yield f"data: {json.dumps(response)}\n\n"
                        
                        elif method == "tools/call":
                            # Call a specific tool
                            tool_name = params.get("name")
                            arguments = params.get("arguments", {})
                            
                            try:
                                result = await handle_call_tool(tool_name, arguments)
                                response = {
                                    "jsonrpc": "2.0", 
                                    "id": request_id,
                                    "result": {
                                        "content": [content.model_dump() for content in result]
                                    }
                                }
                                yield f"data: {json.dumps(response)}\n\n"
                            except Exception as e:
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_id,
                                    "error": {"code": -1, "message": str(e)}
                                }
                                yield f"data: {json.dumps(response)}\n\n"
                        
                        elif method == "initialize":
                            # Initialize the server
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": {
                                    "protocolVersion": "2024-11-05",
                                    "capabilities": {
                                        "tools": {"listChanged": True},
                                        "resources": {},
                                        "prompts": {},
                                        "experimental": {}
                                    },
                                    "serverInfo": {
                                        "name": "playwright-plus-server",
                                        "version": "0.1.0"
                                    }
                                }
                            }
                            yield f"data: {json.dumps(response)}\n\n"
                        
                        else:
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {"code": -32601, "message": f"Method not found: {method}"}
                            }
                            yield f"data: {json.dumps(response)}\n\n"
                    
                    return StreamingResponse(
                        generate_response(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                        }
                    )
                    
                except Exception as e:
                    response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                    }
                    async def error_response():
                        yield f"data: {json.dumps(response)}\n\n"
                    
                    return StreamingResponse(
                        error_response(),
                        media_type="text/event-stream"
                    )
            
            # Add root endpoint for OpenAI compatibility
            @app.post("/")
            async def root_sse_endpoint(request: Request):
                """Handle MCP requests at root path (OpenAI expects this)"""
                return await sse_endpoint(request)
            
            @app.get("/")
            async def root_info():
                """Provide server info at root path"""
                return JSONResponse({
                    "name": "playwright-plus-server",
                    "version": "0.1.0",
                    "protocol": "mcp",
                    "transport": "sse",
                    "endpoints": {
                        "mcp": "/",
                        "sse": "/sse",
                        "health": "/health"
                    }
                })
        
        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            """Handle MCP JSON-RPC requests over HTTP"""
            try:
                data = await request.json()
                
                # Handle different MCP method types
                method = data.get("method")
                params = data.get("params", {})
                request_id = data.get("id")
                
                if method == "tools/list":
                    # Get available tools
                    tools = await handle_list_tools()
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": [tool.model_dump() for tool in tools]}
                    })
                
                elif method == "tools/call":
                    # Call a specific tool
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    
                    try:
                        result = await handle_call_tool(tool_name, arguments)
                        return JSONResponse({
                            "jsonrpc": "2.0", 
                            "id": request_id,
                            "result": {
                                "content": [content.model_dump() for content in result]
                            }
                        })
                    except Exception as e:
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -1, "message": str(e)}
                        })
                
                elif method == "initialize":
                    # Initialize the server
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {"listChanged": True},
                                "resources": {},
                                "prompts": {},
                                "experimental": {}
                            },
                            "serverInfo": {
                                "name": "playwright-plus-server",
                                "version": "0.1.0"
                            }
                        }
                    })
                
                else:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"}
                    })
                    
            except Exception as e:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                })
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "server": "playwright-mcp"}
        
        @app.options("/mcp")
        @app.options("/sse")
        @app.options("/")
        async def options_handler():
            """Handle CORS preflight requests"""
            return JSONResponse(
                content="OK",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        
        transport_type = "SSE" if args.transport == 'sse' else "HTTP"
        endpoint = "/sse" if args.transport == 'sse' else "/mcp"
        
        print(f"🚀 Starting Playwright MCP Server on {transport_type} transport")
        print(f"📡 Server running at: http://{args.host}:{args.port}")
        print(f"🔧 MCP endpoint: http://{args.host}:{args.port}{endpoint}")
        print(f"❤️ Health check: http://{args.host}:{args.port}/health")
        
        # Create server configuration for async serving
        config = uvicorn.Config(
            app=app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
        
    else:
        # Original stdio transport
        print("🚀 Starting Playwright MCP Server on stdio transport")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="playwright-plus-server",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

if __name__ == "__main__":
    asyncio.run(main())
