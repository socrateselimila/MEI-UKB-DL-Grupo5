import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import pickle
import os
from collections import Counter

# ==========================================
# 0. CLASSE SIMPLETFIDTF (NECESSÁRIA PARA O PICKLE)
# ==========================================
def tokenize(text):
    text = str(text).lower()
    return re.findall(r'\b[a-z]{2,}\b', text)

class SimpleTFIDF:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, raw_documents):
        doc_counts = Counter()
        total_docs = len(raw_documents)
        tokenized_docs = [tokenize(doc) for doc in raw_documents]
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_counts[token] += 1
        most_common = doc_counts.most_common(self.max_features)
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        for word, idx in self.vocabulary_.items():
            df = doc_counts[word]
            self.idf_[word] = np.log((total_docs + 1) / (df + 1)) + 1

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

# ==========================================
# 1. ARQUITETURAS DOS MODELOS (PYTORCH E NUMPY)
# ==========================================
class Vocabulary:
    def __init__(self, w2i_dict, pad_token="<PAD>", unk_token="<UNK>"):
        self.w2i = w2i_dict
        self.pad_token = pad_token
        self.unk_token = unk_token

    def encode(self, text, max_len=128):
        tokens = tokenize(text)
        ids = [self.w2i.get(token, self.w2i.get(self.unk_token, 1)) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.w2i.get(self.pad_token, 0)] * (max_len - len(ids))
        return ids

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers, 
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def numpy_dnn_predict(X, params):
    cache_A = X.T
    L = len(params) // 2
    for l in range(1, L):
        W, b = params[f'W{l}'], params[f'b{l}']
        cache_A = relu(np.dot(W, cache_A) + b)
    W_last, b_last = params[f'W{L}'], params[f'b{L}']
    A_last = sigmoid(np.dot(W_last, cache_A) + b_last)
    return A_last.T

# ==========================================
# 2. CONFIGURAÇÃO DE CAMINHOS E GRUPO
# ==========================================
GRUPO_NUMERO = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

INPUT_PATH = os.path.join(ROOT_DIR, 'data', 'dataset3_inputs.csv')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'Submissao1')
MODEL_NUMPY_PATH = os.path.join(ROOT_DIR, 'models', 'numpy_dnn.pkl')
MODEL_PYTORCH_PATH = os.path.join(ROOT_DIR, 'models', 'pytorch_bilstm.pth')

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Dispositivo de execução: {device.type}")

# ==========================================
# 3. CARREGAR MODELOS
# ==========================================
print("A carregar os modelos guardados...")
with open(MODEL_NUMPY_PATH, 'rb') as f:
    numpy_artifacts = pickle.load(f)
vectorizer_numpy = numpy_artifacts['vectorizer']
numpy_params = numpy_artifacts['params']

checkpoint = torch.load(MODEL_PYTORCH_PATH, map_location=device)
vocab_pytorch = Vocabulary(checkpoint['vocab_w2i'])
pytorch_model = TextBiLSTM(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_dim=checkpoint['hidden_dim']
).to(device)
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()
print("Modelos carregados com sucesso!")

# ==========================================
# 4. LER O DATASET COM SEPARADOR CORRETO (;)
# ==========================================
print(f"A ler o dataset de input de: {INPUT_PATH}")
df_test = pd.read_csv(INPUT_PATH, sep=';', engine='python', on_bad_lines='skip')

num_exemplos = len(df_test)
print(f"Total de exemplos lidos com sucesso: {num_exemplos}")
print(f"Total de exemplos a classificar: {num_exemplos}")

# Identificar qual é a coluna de texto ('Text' ou índice 1)
text_col = 'Text' if 'Text' in df_test.columns else df_test.columns[1]
texts = df_test[text_col].astype(str).values

# ==========================================
# 5. INFERÊNCIA E GRAVAÇÃO DOS FICHEIROS
# ==========================================
print("A executar inferência com o Modelo 1 (NumPy DNN)...")
X_test_np = vectorizer_numpy.transform(texts)
probs_np = numpy_dnn_predict(X_test_np, numpy_params).flatten()
preds_np_labels = ["AI" if p >= 0.5 else "Human" for p in probs_np]

df_sub1 = pd.DataFrame({
    'text': texts,
    'output': preds_np_labels
})
path_sub1 = os.path.join(OUTPUT_DIR, f"submissao1-grupo{GRUPO_NUMERO}-s1.csv")
df_sub1.to_csv(path_sub1, index=False)
print(f"Guardado com sucesso: {path_sub1}")

print("A executar inferência com o Modelo 2 (PyTorch Bi-LSTM)...")
encoded_texts = [vocab_pytorch.encode(t, max_len=checkpoint['max_len']) for t in texts]
tensor_x = torch.tensor(encoded_texts, dtype=torch.long).to(device)

with torch.no_grad():
    logits = pytorch_model(tensor_x).squeeze(-1)
    probs_torch = torch.sigmoid(logits).cpu().numpy()

preds_torch_labels = ["AI" if p >= 0.5 else "Human" for p in probs_torch]

df_sub2 = pd.DataFrame({
    'text': texts,
    'output': preds_torch_labels
})
path_sub2 = os.path.join(OUTPUT_DIR, f"submissao1-grupo{GRUPO_NUMERO}-s2.csv")
df_sub2.to_csv(path_sub2, index=False)
print(f"Guardado com sucesso: {path_sub2}")

print("\nProcesso concluído! Os ficheiros de submissão estão prontos na pasta 'Submissao1/'.")