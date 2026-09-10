import os
import json
import re
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from ollama import Client

class ConfiguradorOllama:
    """
    Configura as variáveis de ambiente necessárias para enganar o pacote topicgpt_python,
    fazendo-o redirecionar chamadas da API da OpenAI para o seu servidor Ollama local.
    """
    @staticmethod
    def setup_ambiente_topicgpt(host="127.0.0.1", port="11434"):
        base_url = f"http://{host}:{port}/v1"
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = "ollama-key-falsa" # Obrigatório
        print(f"Ambiente configurado! topicgpt_python apontará para: {base_url}")

class AnalisadorLLM:
    """
    Classe para encapsular a interação com o LLM via Ollama.
    """
    def __init__(self, host: str, model: str):
        print(f"Inicializando cliente LLM para o host '{host}' e modelo '{model}'...")
        self.client = Client(host=host)
        self.model = model

    def chamar_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Erro ao contatar o LLM: {e}")
            return "ERRO_LLM"

class PipelineMestrado:
    """
    Classe para organizar a execução do experimento de mestrado usando topicgpt_python.
    """
    def __init__(self, model_name="gemma2:9b"):
        self.api = "openai" 
        self.model_name = model_name

    def gerar_topicos(self, data_file, prompt_file, seed_file, out_file, topic_file):
        from topicgpt_python import generate_topic_lvl1
        print("Iniciando geração de tópicos (nível 1)...")
        generate_topic_lvl1(
            api=self.api,
            model=self.model_name,
            data=data_file, 
            prompt_file=prompt_file,
            seed_file=seed_file,
            out_file=out_file,
            topic_file=topic_file,
            verbose=True
        )
        print(f"Tópicos gerados e salvos em: {topic_file}")

    def refinar_topicos(self, prompt_file, generation_file, topic_file, out_file, updated_file, mapping_file):
        from topicgpt_python import refine_topics
        print("Refinando tópicos gerados...")
        refine_topics(
            api=self.api,
            model=self.model_name,
            prompt_file=prompt_file,
            generation_file=generation_file,
            topic_file=topic_file,
            out_file=out_file,
            updated_file=updated_file,
            verbose=True,
            remove=True,
            mapping_file=mapping_file
        )
        print("Refinamento concluído.")

    def atribuir_topicos(self, data_file, prompt_file, out_file, topic_file):
        from topicgpt_python import assign_topics
        
        print(f"Iniciando atribuição lendo o dataset: {data_file}")
        print(f"Lendo a árvore de tópicos de: {topic_file}")
        
        assign_topics(
            api=self.api,
            model=self.model_name,
            data=data_file, 
            prompt_file=prompt_file,
            out_file=out_file,
            topic_file=topic_file,
            verbose=True
        )
        print("Atribuição concluída.")

class AvaliadorMetricas:
    """
    Classe para comparar os resultados do modelo (LLM open source) 
    com o ground-truth utilizando métricas de clusterização.
    """
    @staticmethod
    def calcular_purity(y_true, y_pred):
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=0)) / np.sum(matriz)

    @staticmethod
    def calcular_inverse_purity(y_true, y_pred):
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=1)) / np.sum(matriz)

    @staticmethod
    def avaliar_resultados(caminho_arquivo_json):
        y_true = []
        y_pred = []

        with open(caminho_arquivo_json, 'r', encoding='utf-8') as f:
            try:
                dados = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                dados = [json.loads(linha) for linha in f]

        for item in dados:
            verdadeiro = item.get('label_name')
            # O TopicGPT salvou as classificações finais na chave 'responses'
            predito_bruto = item.get('responses', '') 
            
            if verdadeiro is not None and predito_bruto:
                # O LLM é verboso. Precisamos extrair apenas o nome do tópico para as métricas.
                # Exemplo bruto: "Based on... Assignment: [1] Sports: Mentions..."
                # Procuramos o padrão `[X] NomeDoTópico:`
                
                match = re.search(r"Assignment:\s*\[\d+\]\s*([^:]+)", predito_bruto, re.IGNORECASE)
                if not match:
                    # Fallback caso a palavra "Assignment" falte
                    match = re.search(r"\[\d+\]\s*([^:]+)", predito_bruto)
                
                predito_limpo = match.group(1).strip().lower() if match else "indefinido"
                
                # Normaliza o ground truth
                verdadeiro_limpo = str(verdadeiro).strip().lower()

                y_true.append(verdadeiro_limpo)
                y_pred.append(predito_limpo)

        if not y_true:
            print("Erro: Não foram encontrados 'label_name' ou predições no arquivo.")
            return None

        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        purity = AvaliadorMetricas.calcular_purity(y_true, y_pred)
        inv_purity = AvaliadorMetricas.calcular_inverse_purity(y_true, y_pred)
        
        hmp = 0 if (purity + inv_purity) == 0 else 2 * (purity * inv_purity) / (purity + inv_purity)

        print(f"--- Resultados da Avaliação vs Ground-Truth ---")
        print(f"Amostras válidas avaliadas: {len(y_true)}")
        print(f"NMI: {nmi:.4f}")
        print(f"ARI: {ari:.4f}")
        print(f"Purity: {purity:.4f} | Inverse Purity: {inv_purity:.4f}")
        print(f"HMP: {hmp:.4f}")
        print(f"-----------------------------------------------")
        
        return {"NMI": nmi, "ARI": ari, "HMP": hmp, "Purity": purity, "Inverse_Purity": inv_purity}