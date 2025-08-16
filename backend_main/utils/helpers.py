import json5
import re
import numpy as np
from typing import List, Dict, Optional

def extract_json_from_llm(text: str) -> Optional[Dict]:
    """
    Extracts a JSON object or list from a string, robustly handling
    markdown code fences and using json5 for flexibility.
    """
    # Regex to find a JSON object or array within ```json ... ```
    match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    
    json_str = match.group(1) if match else text
    
    try:
        # Use json5 to handle trailing commas, comments, etc.
        return json5.loads(json_str)
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}. String was: {json_str}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates the cosine similarity between two vectors.
    Returns 0 if one of the vectors has a magnitude of 0.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
        
    return dot_product / norm_product
