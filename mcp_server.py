from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, AnyUrl, Field
import readabilipy
from pathlib import Path
import asyncio
import os

# Additional imports for enhanced resume tool
import PyPDF2
from docx import Document

# Load configuration from environment variables
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

TOKEN = os.getenv("TOKEN")  # Get API token from environment variables
MY_NUMBER = os.getenv("MY_NUMBER")  # Get phone number from environment variables (format: country_code + number)

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    """
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,  # No expiration for simplicity
            )
        return None

class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
        """
        import httpx
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except Exception as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )

        if response.status_code >= 400:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch {url} - status code {response.status_code}",
                )
            )

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
        try:
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
        except Exception as e:
            return f"<e>Error processing HTML: {str(e)}</e>"

# Initialize the MCP server
mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)

# ENHANCED RESUME TOOL - Supports .txt, .md, .pdf, and .docx files
@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    Supports .txt, .md, .pdf, and .docx files.
    """
    try:
        # Look for resume files in common formats
        resume_files = []
        current_dir = Path(".")
        
        # Search for resume files
        for pattern in ["resume.*", "cv.*", "Resume.*", "CV.*"]:
            resume_files.extend(current_dir.glob(pattern))
        
        if not resume_files:
            return "# Resume\n\nNo resume file found. Please place your resume file in the current directory with a name like 'resume.pdf', 'resume.docx', 'resume.txt', or 'resume.md'."
        
        # Use the first resume file found
        resume_file = resume_files[0]
        
        if resume_file.suffix.lower() == '.txt':
            # Read plain text file
            with open(resume_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        
        elif resume_file.suffix.lower() == '.md':
            # Read markdown file
            with open(resume_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        
        elif resume_file.suffix.lower() == '.pdf':
            # Read PDF file
            try:
                with open(resume_file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return f"# Resume\n\n{text.strip()}"
            except Exception as e:
                return f"# Resume\n\nError reading PDF file: {str(e)}\n\nPlease ensure the PDF is not password protected."
        
        elif resume_file.suffix.lower() in ['.docx', '.doc']:
            # Read Word document
            try:
                doc = Document(resume_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return f"# Resume\n\n{text.strip()}"
            except Exception as e:
                return f"# Resume\n\nError reading Word document: {str(e)}"
        
        else:
            return f"# Resume\n\nUnsupported file format: {resume_file.suffix}\n\nSupported formats: .txt, .md, .pdf, .docx"
            
    except Exception as e:
        return f"# Resume\n\nError reading resume file: {str(e)}\n\nPlease ensure your resume file is accessible and in a supported format."

@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)

@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    
    if start_index >= original_length:
        content = "<e>No more content available.</e>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<e>No more content available.</e>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            
            # Only add the prompt to continue fetching if there is still remaining content
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<e>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</e>"

    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]

async def main():
    print(f"Starting MCP server on http://0.0.0.0:8085")
    print(f"Auth token: {TOKEN}")
    print(f"Phone number: {MY_NUMBER}")
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )

if __name__ == "__main__":
    asyncio.run(main())