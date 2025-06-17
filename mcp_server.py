import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import markdownify
import readabilipy
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
import docx
import httpx
import uvicorn

# REPLACE THESE WITH YOUR ACTUAL VALUES
TOKEN = "1d1bd2ed584e"  # Replace with your actual application key
MY_NUMBER = "917620957998"  # Replace with your phone number (e.g., 919876543210)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Puch MCP Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MCPMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# Authentication
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = authorization.split(" ")[1]
    if token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return token

class ResumeProcessor:
    """Handle resume file processing and conversion to markdown."""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")

    @staticmethod
    def read_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {e}")

    @staticmethod
    def read_txt(file_path: str) -> str:
        """Read plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Failed to read text file: {e}")

    @staticmethod
    def convert_to_markdown(text: str) -> str:
        """Convert text to markdown format with basic formatting."""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue
                
            # Simple heuristics for markdown conversion
            if line.isupper() and len(line) > 3:
                # Likely a section header
                markdown_lines.append(f"## {line.title()}")
            elif line.endswith(':') and len(line.split()) <= 5:
                # Likely a subsection
                markdown_lines.append(f"### {line}")
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)

    @classmethod
    def get_resume_content(cls) -> str:
        """Find and process resume file, return as markdown."""
        try:
            # Look for resume files in common locations and formats
            possible_paths = [
                "resume.pdf",
                "resume.docx", 
                "resume.txt",
                "cv.pdf",
                "cv.docx",
                "cv.txt",
                "./resume.pdf",
                "./resume.docx",
                "./resume.txt"
            ]
            
            resume_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    resume_path = path
                    break
            
            if not resume_path:
                return """# Resume Not Found

Please place your resume file in the server directory with one of these names:
- resume.pdf
- resume.docx  
- resume.txt
- cv.pdf
- cv.docx
- cv.txt
"""
            
            # Read the file based on its extension
            file_extension = Path(resume_path).suffix.lower()
            
            if file_extension == '.pdf':
                text_content = cls.read_pdf(resume_path)
            elif file_extension == '.docx':
                text_content = cls.read_docx(resume_path)
            elif file_extension == '.txt':
                text_content = cls.read_txt(resume_path)
            else:
                raise Exception(f"Unsupported file format: {file_extension}")
            
            # Convert to markdown
            markdown_content = cls.convert_to_markdown(text_content)
            
            return markdown_content
            
        except Exception as e:
            return f"# Error Processing Resume\n\nFailed to process resume: {str(e)}"

class WebFetcher:
    """Handle web content fetching and processing."""
    
    USER_AGENT = "Puch/1.0 (Autonomous)"
    
    @classmethod
    async def fetch_url(cls, url: str, force_raw: bool = False) -> tuple[str, str]:
        """Fetch URL and return content ready for LLM."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": cls.USER_AGENT},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise Exception(f"Failed to fetch {url}: {e}")

        if response.status_code >= 400:
            raise Exception(f"Failed to fetch {url} - status code {response.status_code}")

        page_raw = response.text
        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<e>Page failed to be simplified from HTML</e>"

        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content

# MCP Tool implementations
class MCPTools:
    @staticmethod
    async def list_tools():
        """Return available tools."""
        return {
            "tools": [
                {
                    "name": "resume",
                    "description": "Return your resume in markdown format. Use when asked for resume or CV.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "validate",
                    "description": "Validate phone number for Puch AI. This tool must be present.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "fetch",
                    "description": "Fetch a URL and return its content in markdown format.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to fetch"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum number of characters to return",
                                "default": 5000,
                                "minimum": 1,
                                "maximum": 1000000
                            },
                            "start_index": {
                                "type": "integer",
                                "description": "Starting character index for truncated content",
                                "default": 0,
                                "minimum": 0
                            },
                            "raw": {
                                "type": "boolean",
                                "description": "Return raw HTML instead of simplified markdown",
                                "default": False
                            }
                        },
                        "required": ["url"]
                    }
                }
            ]
        }
    
    @staticmethod
    async def call_tool(name: str, arguments: Dict[str, Any]):
        """Handle tool calls."""
        try:
            if name == "resume":
                content = ResumeProcessor.get_resume_content()
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                }
            
            elif name == "validate":
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": MY_NUMBER
                        }
                    ]
                }
            
            elif name == "fetch":
                url = arguments.get("url")
                if not url:
                    raise ValueError("URL is required")
                
                max_length = arguments.get("max_length", 5000)
                start_index = arguments.get("start_index", 0)
                raw = arguments.get("raw", False)
                
                content, prefix = await WebFetcher.fetch_url(url, force_raw=raw)
                original_length = len(content)
                
                if start_index >= original_length:
                    content = "<e>No more content available.</e>"
                else:
                    truncated_content = content[start_index:start_index + max_length]
                    if not truncated_content:
                        content = "<e>No more content available.</e>"
                    else:
                        content = truncated_content
                        
                    actual_content_length = len(truncated_content)
                    remaining_content = original_length - (start_index + actual_content_length)
                    
                    if actual_content_length == max_length and remaining_content > 0:
                        next_start = start_index + actual_content_length
                        content += f"\n\n<e>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</e>"
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prefix}Contents of {url}:\n{content}"
                        }
                    ]
                }
            
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Error in tool {name}: {e}")
            raise HTTPException(status_code=500, detail=f"Tool error: {str(e)}")

# Routes
@app.get("/")
async def root():
    return {"message": "Puch MCP Server is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/mcp")
async def mcp_handler(request: Request, token: str = Depends(verify_token)):
    """Main MCP endpoint handler."""
    try:
        body = await request.json()
        message = MCPMessage(**body)
        
        logger.info(f"Received MCP request: {message.method}")
        
        if message.method == "tools/list":
            result = await MCPTools.list_tools()
            return MCPMessage(
                id=message.id,
                result=result
            )
        
        elif message.method == "tools/call":
            if not message.params:
                raise HTTPException(status_code=400, detail="Missing parameters")
            
            tool_name = message.params.get("name")
            arguments = message.params.get("arguments", {})
            
            result = await MCPTools.call_tool(tool_name, arguments)
            return MCPMessage(
                id=message.id,
                result=result
            )
        
        elif message.method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "puch-mcp-server",
                    "version": "1.0.0"
                }
            }
            return MCPMessage(
                id=message.id,
                result=result
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {message.method}")
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"MCP handler error: {e}")
        return MCPMessage(
            id=getattr(message, 'id', None),
            error={
                "code": -32603,
                "message": str(e)
            }
        )

# Simple tool endpoints for testing
@app.get("/test/resume")
async def test_resume():
    """Test endpoint for resume tool."""
    content = ResumeProcessor.get_resume_content()
    return {"resume": content}

@app.get("/test/validate")
async def test_validate():
    """Test endpoint for validate tool."""
    return {"phone_number": MY_NUMBER}

if __name__ == "__main__":
    logger.info("Starting Puch MCP HTTP Server...")
    logger.info(f"Validation number: {MY_NUMBER}")
    logger.info(f"Token configured: {'Yes' if TOKEN != 'YOUR_APPLICATION_KEY_HERE' else 'No - PLEASE UPDATE TOKEN'}")
    
    # Check if resume file exists
    resume_files = ["resume.pdf", "resume.docx", "resume.txt", "cv.pdf", "cv.docx", "cv.txt"]
    found_resume = any(os.path.exists(f) for f in resume_files)
    if found_resume:
        logger.info("Resume file found and ready")
    else:
        logger.warning("No resume file found. Please add one of: " + ", ".join(resume_files))
    
    uvicorn.run(app, host="0.0.0.0", port=8085)