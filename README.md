# MEI-UKB-DL-Grupo5
Projecto Prático da UC de Aprendizagem Profunda (2026) - Detecção de Texto Gerado por IA vs Humano. MEI - Universidade Katyavala Bwila

# Detecção de Texto Gerado por IA vs. Humano
**Unidade Curricular:** Aprendizagem Profunda (Deep Learning) — 2026  
**Mestrado em Engenharia Informática — Universidade Katyavala Bwila**  
**Docentes:** Prof. Miguel Rocha e Prof. Lázaro Makili  

---

## 👥 Constituição do Grupo de Trabalho
* **Alfredo Sócrates Elimila 1** — [alfredosocrateselimila@gmail.com / socrateselimila]
* **Cardoso da Silva Monteiro 2** — [Email / GitHub]
* **Eunice Patrícia de Freitas Barros 3** — [929532160.eunice@gmail.com / GitHub]
* **Rita Tavita Carlos Elias  4** — [Email / GitHub]

---

## 🎯 Objectivo do Projecto
O objectivo principal deste projecto é desenvolver, treinar e avaliar modelos de *Machine Learning* e *Deep Learning* capazes de classificar pequenos textos em língua inglesa em duas classes (classificação binária):
* `ai`: Texto gerado por modelos/aplicações de Inteligência Artificial (ex: Chatbots, LLMs).
* `human`: Texto escrito por seres humanos.

---

## 📂 Estrutura do Repositório
```text
├── data/                  # Datasets brutos e processados (inputs, outputs)
├── notebooks/             # Notebooks Jupyter de exploração, treino e validação
│   ├── 01_data_prep.ipynb # Construção e limpeza de datasets
│   ├── 02_numpy_models.ipynb # Modelos implementados de raiz (NumPy)
│   └── 03_pytorch_models.ipynb # Modelos avançados em PyTorch
├── models/                # Modelos treinados salvos (.pkl, .pt, .pth)
├── src/                   # Módulos Python com rotinas reutilizáveis de código
├── reports/               # Relatório final em PDF (máximo 10 páginas)
├── submission_notebook.ipynb # Notebook oficial para inferência e avaliação
└── README.md              # Apresentação geral do repositório
```

---

## 🛠️ Recursos, Ferramentas e Arquitecturas
* **Linguagem de Programação:** Python 3.10+
* **Ambiente de Desenvolcimento:** Jupyter Notebook / VS Code / Google Colab
* **Bibliotecas Principais:** `NumPy`, `Pandas`, `PyTorch`, `Matplotlib`, `NLTK` / `Scikit-learn` (apenas para pré-processamento/TF-IDF)
* **Engenharia de Features:** Bag-of-Words, TF-IDF, Word Embeddings (Word2Vec / GloVe / PyTorch Embedding Layer).
* **Funções de Activação:** ReLU, Sigmoid, LeakyReLU.
* **Técnicas de Regularização e Otimização:** Dropout, Regularização $L_2$ (Weight Decay), Early Stopping, Adam / SGD.
* **Tipos de Redes Neurais Exploradas:**
  1. *Baseline & Raiz (NumPy):* Logistic Regression e Deep Neural Networks (DNNs) tabulares.
  2. *PyTorch:* Feedforward DNNs, Recurrent Neural Networks (RNN, LSTM, GRU Bidirecionais), Transformers / BERT.

---

## 🗓️ Etapas de Desenvolvimento
- [x] **Etapa 1:** Criação do repositório e estruturação inicial.
- [ ] **Etapa 2:** Recolha, limpeza e fusão dos datasets de treino (Tarefa 1).
- [ ] **Etapa 3:** Implementação dos modelos de raiz em NumPy (Tarefa 2).
- [ ] **Etapa 4:** Desenvolvimento e ajuste de hiperparâmetros dos modelos PyTorch (Tarefa 3).
- [ ] **Etapa 5:** Preparação dos pipelines de teste e 1ª Submissão dos modelos (Tarefa 4).
- [ ] **Etapa 6:** Redação do relatório final em PDF e preparação da defesa (Tarefa 5 & 6).