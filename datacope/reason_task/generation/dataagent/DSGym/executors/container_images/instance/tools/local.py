import requests

# Configuration
RETRIEVAL_SERVER_URL = "http://host.docker.internal:8000"  # Points to the host machine's port 8000

def search(query, top_k=3, return_scores=True, format_results=True):
    """Search function that connects to the retrieval server"""
    try:
        response = requests.post(
            f"{RETRIEVAL_SERVER_URL}/retrieve",
            json={"queries": [query], "topk": top_k, "return_scores": return_scores}
        )
        
        if response.status_code == 200:
            data = response.json()
            return format_search_results(data) if format_results else data
        else:
            return {"error": f"Retrieval server returned status code {response.status_code}", "message": response.text}
    except Exception as e:
        return {"error": "Failed to connect to retrieval server", "message": str(e)}

def format_search_results(raw_results):
    """Format the raw results into a simple numbered list"""
    formatted_results = []
    
    try:
        # Handle the nested structure of the results
        if "result" in raw_results and len(raw_results["result"]) > 0:
            results = raw_results["result"][0]  # Get the first list of results
            
            for i, item in enumerate(results, 1):
                content = item["document"]["contents"].strip()
                # Format each result as a simple numbered entry with just the content
                entry = f"[{i}] {content}"
                formatted_results.append(entry)
                
        return "\n\n".join(formatted_results)
    except Exception as e:
        return {"error": "Failed to format results", "message": str(e), "raw_results": raw_results} 