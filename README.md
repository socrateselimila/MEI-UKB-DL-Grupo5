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
├── README.md                          # Apresentação do grupo e navegação do projecto
├── data/
│   ├── raw/                           # Datasets originais (dataset1, HC3, etc.)
│   └── processed/                     # Datasets limpos e prontos para treino
├── src/                               # Módulos Python reutilizáveis (.py)
│   ├── dataset_loader.py              # Script para processar e fundir dados
│   ├── preprocessing.py               # Limpeza de texto e vetorização (TF-IDF/BoW)
│   ├── custom_nn.py                   # Implementação Scratch em NumPy (Tarefa 2)
│   └── pytorch_models.py              # Classes dos modelos PyTorch (Tarefa 3)
├── notebooks/                         # Análise exploratória e treinos (.ipynb)
│   ├── 01_eda_and_data_fusion.ipynb   # Exploração e junção dos datasets
│   ├── 02_numpy_baseline.ipynb        # Treino/Validação dos modelos em NumPy
│   ├── 03_pytorch_models.ipynb        # Treino das redes PyTorch (LSTM/Transformers)
│   └── 04_submission_pipeline.ipynb   # Notebook final de inferência para o professor
├── models/                            # Pesos dos modelos guardados (.pth, .pkl)
└── report/
    └── Relatorio_Final.pdf            # Relatório em PDF (máx. 10 páginas)
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
- [x] **Etapa 2:** Recolha, limpeza e fusão dos datasets de treino (Tarefa 1).
- [x] **Etapa 3:** Implementação dos modelos de raiz em NumPy (Tarefa 2).
- [ ] **Etapa 4:** Desenvolvimento e ajuste de hiperparâmetros dos modelos PyTorch (Tarefa 3).
- [ ] **Etapa 5:** Preparação dos pipelines de teste e 1ª Submissão dos modelos (Tarefa 4).
- [ ] **Etapa 6:** Redação do relatório final em PDF e preparação da defesa (Tarefa 5 & 6).