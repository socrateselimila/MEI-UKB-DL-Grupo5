import numpy as np
import re
from collections import Counter

def tokenize(text):
    """Limpa o texto e divide em palavras/tokens."""
    text = str(text).lower()
    tokens = re.findall(r'\b[a-z]{2,}\b', text) # Apenas palavras com 2+ letras
    return tokens

class SimpleTFIDF:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, raw_documents):
        # 1. Contar frequência de documentos para cada palavra
        doc_counts = Counter()
        total_docs = len(raw_documents)
        
        tokenized_docs = [tokenize(doc) for doc in raw_documents]
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_counts[token] += 1
                
        # 2. Selecionar as 'max_features' palavras mais frequentes
        most_common = doc_counts.most_common(self.max_features)
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        
        # 3. Calcular IDF: log(N / df)
        for word, idx in self.vocabulary_.items():
            df = doc_counts[word]
            self.idf_[word] = math.log((total_docs + 1) / (df + 1)) + 1

    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.vocabulary_)))
        for i, doc in enumerate(raw_documents):
            tokens = tokenize(doc)
            tf_counts = Counter(tokens)
            doc_len = len(tokens) if len(tokens) > 0 else 1
            
            for token, count in tf_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf = count / doc_len
                    X[i, idx] = tf * self.idf_[token]
        return X

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)