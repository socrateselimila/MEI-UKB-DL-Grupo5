import pandas as pd
import os # Adicionado para manipulação de caminhos e verificação de arquivos locais

def load_docente_data(inputs_path, outputs_path):
    df_in = pd.read_csv(inputs_path, sep='\t') # Especificar delimitador como tab
    df_out = pd.read_csv(outputs_path, sep='\t') # Corrigido para especificar delimitador como tab
    # Usar pd.merge para combinar os dataframes com base na coluna 'ID'
    df = pd.merge(df_in, df_out, on='ID', how='inner')
    # Ajustar para nomes genéricos
    # ATENÇÃO: As colunas 'Text' e 'Label' foram identificadas como as correctas nos arquivos do docente.
    df = df.rename(columns={'Text': 'text', 'Label': 'label'}) # Corrigido para 'Text' e 'Label'
    return df

def load_external_hc3():
    print("Carregando dados do HC3 localmente de 'data/raw/all.jsonl'")

    try:
        # Carregar o arquivo all.jsonl baixado localmente.
        hc3_data_path = "data/raw/all.jsonl"
        if os.path.exists(hc3_data_path):
            print(f"Carregando {hc3_data_path}...")
            # O arquivo é um JSONL (JSON Lines), então usamos lines=True
            external_df = pd.read_json(hc3_data_path, lines=True)
            # Processar o df externo para o formato desejado (text, label)
            data = []
            for _, row in external_df.iterrows():
                if 'human_answers' in row:
                    for h_ans in row['human_answers']:
                        data.append({'text': h_ans, 'label': 0})
                if 'chatgpt_answers' in row:
                    for ai_ans in row['chatgpt_answers']:
                        data.append({'text': ai_ans, 'label': 1})
            return pd.DataFrame(data)
        else:
            print(f"Arquivo {hc3_data_path} não encontrado. Por favor, certifique-se de que o dataset HC3 (all.jsonl) foi baixado e colocado em 'data/raw/'.")
            print("Retornando DataFrame vazio para HC3.")
            return pd.DataFrame(columns=['text', 'label'])
    except Exception as e:
        print(f"Erro ao carregar ou processar o dataset HC3 localmente: {e}")
        print("Retornando DataFrame vazio para HC3.")
        return pd.DataFrame(columns=['text', 'label'])

def prepare_combined_dataset(inputs_path, outputs_path, save_output_path):
    df_docente = load_docente_data(inputs_path, outputs_path)
    df_external = load_external_hc3()

    # Concatenar e remover duplicados/nulos
    df_full = pd.concat([df_docente, df_external], ignore_index=True)
    df_full = df_full.dropna().drop_duplicates(subset=['text'])

    # Filtrar aproximadamente pelo tamanho indicado pelo professor (opcional/recomendado)
    df_full['word_count'] = df_full['text'].apply(lambda x: len(x.split()))
    # Mantém textos com tamanho razoável para a tarefa
    df_full = df_full[(df_full['word_count'] >= 30) & (df_full['word_count'] <= 300)]

    df_full[['text', 'label']].to_csv(save_output_path, index=False)
    print(f"Dataset final salvo com {len(df_full)} amostras em {save_output_path}")

if __name__ == "__main__":
    # Certifique-se de que o diretório 'data/processed' existe
    os.makedirs("data/processed", exist_ok=True)
    prepare_combined_dataset("data/raw/dataset1_inputs.csv", "data/raw/dataset1_outputs.csv", "data/processed/full_dataset.csv")