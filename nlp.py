import spacy
from typing import List

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "Run: python -m spacy download en_core_web_sm"
    )
    
def tokenize_text(raw_text: str) -> List[str]:
    if not raw_text or not raw_text.strip():
        return []
 
    doc = nlp(raw_text)
 
    cleaned_tokens = [
        token.text.lower()
        for token in doc
        if not token.is_stop        
        and not token.is_punct     
        and not token.is_space      
        and token.text.strip()      
    ]
 
    return cleaned_tokens
 
 
if __name__ == "__main__":
    sample = """
    This is a sample text to demonstrate the tokenization process using spaCy. 
    Random fact: giraffes are about 30 times more likely to be hit by lightning than people.
    """
 
    tokens = tokenize_text(sample)
    print(f"Token count : {len(tokens)}")
    print(f"Tokens      : {tokens}")