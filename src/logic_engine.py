import re

class SymbolicGrounding:
    def __init__(self):
        # Swiss Law KB
        self.kb_patterns = {
            "OR_55": r"OR (Art\.|Artikel) 55",
            "ZGB_2": r"ZGB (Art\.|Artikel) 2",
            "BGE": r"BGE \d+ [IV|V|VI]+ \d+",
            "statute": r"(OR|ZGB|StGB|BV) (Art\.|Artikel) \d+",
            "case_law": r"BGE \d+ [IV|V|VI]+ \d+",
            "doctrine": r"(\w+, \d{4})|(\w+ et al\.)"
        }

    def analyze_claims(self, text):
        """
        Identifies 'Statements' vs 'Evidence'
        Returns a ratio of Groundedness.
        """
        # 1. Identify Statements (Simplified: Sentences ending in period without citation)
        sentences = re.split(r'(?<=[.!?]) +', text)
        statements = [s for s in sentences if len(s) > 20] # Ignore short fragments
        
        # 2. Identify Evidence (Anchors)
        evidence_found = []
        for category, pattern in self.evidence_markers.items():
            matches = re.findall(pattern, text)
            evidence_found.extend(matches)

        # 3. Grounding Logic: 
        # A 'Healthy' Evaluation Brief requires at least 1 evidence anchor per 2 statements.
        if len(statements) == 0: return 1.0
        
        grounding_ratio = len(evidence_found) / (len(statements) / 2)
        return np.clip(grounding_ratio, 0.0, 1.0)
    def get_logic_likelihood(self, text):
        # The Grounding Function G: Text -> Likelihood
        found_count = 0
        for key, pattern in self.kb_patterns.items():
            if re.search(pattern, text):
                found_count += 1
        
        # If no citations are found in a paragraph, 
        # the likelihood of "High Logical Cogency" is low.
        return (found_count + 0.1) / (len(self.kb_patterns))