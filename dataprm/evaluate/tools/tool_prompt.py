import os

QUERY_DOCUMENT_PROMPT = """
def query_document(query: str, file_path: str) -> str:
    '''
    Ask a question about the document
    
    Parameters:
      - query: The question to ask (string)
      - file_path: The path to the document file (supports .txt, .md, etc.)
    
    Returns: The query result as a string
    
    Example:
      result = query_document("What is the calculation formula for the cost?", "manual.md")
    '''
"""

QUERY_IMAGE_PROMPT = """
def query_image(query: str, image_path: str) -> str:
    '''
    Ask a question about the image
    
    Parameters:
      - query: The question to ask (string)
      - image_path: The path to the image file (supports .png, .jpg, etc.)
    
    Returns: The query result as a string
    
    Example:
      result = query_image("Describe the image in detail.", "diagram.png")
    '''
"""

# get the absolute path of the query_document.py file
QUERY_DOCUMENT_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "query_document.py"))

QUERY_IMAGE_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "query_image.py"))