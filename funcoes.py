# -*- coding: utf-8 -*-

"""
Esqueleto de código para o experimento de mestrado com RAG para geração e refinamento de tópicos.
"""

# --- Importações de Bibliotecas ---
import os
import re
from typing import List, Dict, Any, Tuple

# Para interagir com o LLM via Ollama
from ollama import Client

# Para carregar o dataset
from datasets import load_dataset

# Para embeddings e vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Para resultados em CSV e cálculo de métricas
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from collections import Counter, defaultdict

# Para o pré-processamento de texto (mantido caso precise de alguma utilidade do gensim)
import gensim

# --- Constantes de Configuração ---
#OLLAMA_HOST = "http://164.41.75.221:11434"  # Host Ollama, conforme fornecido
OLLAMA_HOST = '127.0.0.1:11434'  # Host Ollama, conforme fornecido
LLM_MODEL = "llama3.1" # Modelo LLM a ser usado no experimento, conforme solicitado
DATASET_NAME = "cardiffnlp/tweet_topic_single"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Modelo de embedding eficiente para a tarefa
NUM_RANDOM_SAMPLES_FOR_LLM = 500  # Quantidade de documentos para gerar tópicos com o LLM
TOP_K_SIMILAR = 3 # Número de documentos similares a serem recuperados para o contexto

# --- Definição das Classes ---

class AnalisadorLLM:
    """
    Classe para encapsular a interação com o Large Language Model (LLM) via Ollama.
    """
    def __init__(self, host: str, model: str):
        """
        Inicializa o cliente Ollama.
        :param host: URL do host onde o Ollama está rodando.
        :param model: Nome do modelo a ser utilizado (ex: 'llama4').
        """
        print(f"Inicializando cliente LLM para o host '{host}' e modelo '{model}'...")
        self.client = Client(host=host)
        self.model = model

    def _criar_prompt_geracao_topico(self, texto_documento: str) -> str:
        """Helper para criar o prompt formatado para o LLM."""
        prompt = f"""
        Analise o seguinte documento e gere um único tópico principal que o descreva.
        O tópico deve ser uma ou duas palavras, como "Business", "Technology", "Health", "Sports" ou "Politics".
        Responda apenas com o tópico e nada mais. Use o mesmo nome para tópicos com sinônimos, por exemplo: 
        Observações com as classificações Famosos e Celebridades não devem ser diferentes, e sim só uma como "Celebridades" por exemplo.

        Documento: "{texto_documento}"

        Tópico:
        """
        return prompt.strip()

    def _criar_prompt_atualizacao_topico(self, topico_inicial: str, documentos_contexto: List[str]) -> str:
        """Helper para criar o prompt de refinamento de tópico com base em contexto."""
        contexto_str = "\n\n".join([f"Documento similar {i+1}:\n\"{doc}\"" for i, doc in enumerate(documentos_contexto)])
        
        prompt = f"""
        Você está refinando a classificação de tópicos.
        O tópico inicial proposto foi: "{topico_inicial}".

        Abaixo estão alguns documentos semanticamente similares.
        Analise este contexto adicional para confirmar ou refinar o tópico inicial.
        Mantenha o mesmo padrão de tópicos da primeira classificação e não os deixe mais precisos,
        essa etapa serve para confirmar o tópico ou escolher outro.

        Contexto dos documentos similares:
        {contexto_str}

        Com base neste contexto, qual é o tópico refinado ou confirmado? Responda apenas com o tópico.

        Tópico Refinado:
        """
        return prompt.strip()


    def gerar_topico_para_documento(self, texto_documento: str) -> str:
        """
        Envia um documento para o LLM e retorna o tópico gerado.
        """
        prompt = self._criar_prompt_geracao_topico(texto_documento)
        
        try:
            print(f"\nEnviando documento para o LLM:\n'{texto_documento[:100]}...'")
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            topico = response['message']['content'].strip()
            print(f"LLM respondeu com o tópico inicial: '{topico}'")
            return topico
        except Exception as e:
            print(f"Erro ao contatar o LLM: {e}")
            return "ERRO_LLM"

    def atualizar_topico_com_contexto(self, topico_inicial: str, documentos_contexto: List[str]) -> str:
        """Envia o tópico inicial e um contexto de documentos similares para o LLM refinar o tópico."""
        prompt = self._criar_prompt_atualizacao_topico(topico_inicial, documentos_contexto)
        try:
            print(f"Enviando contexto para refinar o tópico '{topico_inicial}'...")
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            topico_atualizado = response['message']['content'].strip()
            print(f"LLM refinou para o tópico: '{topico_atualizado}'")
            return topico_atualizado
        except Exception as e:
            print(f"Erro ao contatar o LLM para refinamento: {e}")
            return "ERRO_REFINAMENTO_LLM"

class VectorStore:
    """
    Classe para gerenciar a criação de embeddings e a busca por similaridade com FAISS.
    """
    def __init__(self, model_name: str):
        print(f"Carregando modelo de embedding '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.dimension = self.model.get_sentence_embedding_dimension()

    def create_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Gera embeddings para uma lista de textos."""
        print(f"Gerando embeddings para {len(texts)} textos...")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=show_progress)
        return np.array(embeddings).astype('float32')

    def build_index(self, embeddings: np.ndarray):
        """Constrói um índice FAISS a partir dos embeddings."""
        print(f"Construindo índice FAISS para {len(embeddings)} vetores de dimensão {self.dimension}...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print("Índice construído com sucesso.")

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Busca os k vizinhos mais próximos de um embedding de consulta."""
        if self.index is None:
            raise RuntimeError("O índice não foi construído. Chame build_index() primeiro.")
        
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    
# --- Funções de Avaliação de Tópicos ---

def preprocess_text(text: str, stop_words: set) -> List[str]:
    """
    Tokeniza, remove stopwords, pontuação e palavras curtas.
    """
    # simple_preprocess faz a tokenização e passa para minúsculo
    return [word for word in simple_preprocess(text) if word not in stop_words and len(word) > 2]

def extract_top_n_words(topic_docs: Dict[str, List[str]], stop_words: set, top_n: int = 10) -> Dict[str, List[str]]:
    """
    Extrai as N palavras mais frequentes para cada tópico (grupo de documentos).
    Isso transforma as classificações do LLM (ex: "Sports") em um tópico 
    tradicional (ex: ["game", "team", "play", ...]).
    """
    print("Extraindo top N palavras para cada tópico gerado...")
    topics_with_words = {}
    for topic, docs in topic_docs.items():
        if not docs:
            continue
        
        # Juntar todos os documentos do tópico e processar
        full_text = " ".join(docs)
        tokens = preprocess_text(full_text, stop_words)
        
        # Contar as palavras mais comuns
        word_counts = Counter(tokens)
        top_words = [word for word, _ in word_counts.most_common(top_n)]
        topics_with_words[topic] = top_words
    
    print(f"Top palavras extraídas: {topics_with_words}")
    return topics_with_words

def calculate_topic_coherence(topics_with_words: Dict[str, List[str]], documents: List[str], stop_words: set, coherence_type: str = 'c_v') -> float:
    """
    Calcula a coerência (ex: C_v) para um conjunto de tópicos e documentos.
    'documents' deve ser a lista de textos usados para construir os tópicos.
    """
    print(f"Calculando Topic Coherence ({coherence_type})...")
    
    # 1. Processar os documentos de referência
    processed_docs = [preprocess_text(doc, stop_words) for doc in documents]
    
    # 2. Criar dicionário Gensim
    dictionary = Dictionary(processed_docs)
    
    # 3. Formatar os tópicos (lista de listas de palavras)
    topics_list = list(topics_with_words.values())
    
    # 4. Calcular coerência
    if not topics_list or not dictionary or not processed_docs:
        print("Não foi possível calcular a coerência (tópicos, dicionário ou documentos vazios).")
        return 0.0

    try:
        coherence_model = CoherenceModel(
            topics=topics_list,
            texts=processed_docs,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        coherence = coherence_model.get_coherence()
        print(f"Coerência calculada: {coherence}")
        return coherence
    except Exception as e:
        print(f"Erro ao calcular coerência: {e}")
        return 0.0

def calculate_topic_diversity(topics_with_words: Dict[str, List[str]]) -> float:
    """
    Calcula a diversidade (proporção de palavras únicas) entre as top N palavras
    de todos os tópicos.
    """
    print("Calculando Topic Diversity...")
    if not topics_with_words:
        return 0.0

    all_words = []
    for words in topics_with_words.values():
        all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    unique_words = set(all_words)
    diversity = len(unique_words) / len(all_words)
    print(f"Diversidade calculada: {diversity}")
    return diversity


# --- Função Principal de Execução do Experimento ---
def main():
    """
    Orquestra a execução do experimento.
    """
    # 1. Carregamento do Dataset
    print(f"Carregando dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split='train_all')
    documentos = dataset['text']
    print(f"{len(documentos)} documentos carregados.")

    # (NOVO) Download de recursos NLTK (stopwords)
    print("Baixando stopwords do NLTK...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # (FIM DO NOVO)

    # 2. Inicialização dos componentes
    analisador_llm = AnalisadorLLM(host=OLLAMA_HOST, model=LLM_MODEL)
    vector_store = VectorStore(model_name=EMBEDDING_MODEL_NAME)
    
    # 3. Construção do Vector Store da base completa
    corpus_embeddings = vector_store.create_embeddings(documentos, show_progress=True)
    vector_store.build_index(corpus_embeddings)
    
    # 4. Análise com LLM e RAG (em uma amostra)
    print(f"\n--- INICIANDO PROCESSO DE RAG PARA {NUM_RANDOM_SAMPLES_FOR_LLM} AMOSTRAS ---")
    amostra_docs = dataset.shuffle(seed=42).select(range(NUM_RANDOM_SAMPLES_FOR_LLM))

    resultados_finais = [] # (NOVO) Lista para salvar os resultados
    
    for i, doc in enumerate(amostra_docs):
        print(f"\n{'='*20} Processando Amostra {i+1}/{NUM_RANDOM_SAMPLES_FOR_LLM} {'='*20}")
        texto_original = doc['text']
        
        # Etapa 1: Gerar tópico inicial para o documento da amostra.
        topico_inicial = analisador_llm.gerar_topico_para_documento(texto_original)
        if "ERRO" in topico_inicial:
            continue
            
        # Etapa 2: Combinar tópico e documento para criar uma consulta rica para a busca.
        query_text = f"Tópico: {topico_inicial}. Documento: {texto_original}"
        query_embedding = vector_store.create_embeddings([query_text], show_progress=False)
        
        # Etapa 3: Buscar os documentos mais similares na base completa.
        print(f"Buscando {TOP_K_SIMILAR} documentos similares para a consulta...")
        _, similar_indices = vector_store.search(query_embedding, k=TOP_K_SIMILAR)
        
        # Recupera os textos dos documentos encontrados para usar como contexto.
        documentos_contexto = [documentos[idx] for idx in similar_indices[0]]
        
        # Etapa 4: Pedir ao LLM para atualizar o tópico usando o contexto recuperado.
        topico_atualizado = analisador_llm.atualizar_topico_com_contexto(topico_inicial, documentos_contexto)

        # Exibir resultados consolidados para esta amostra
        print("\n--- RESULTADO DA AMOSTRA ---")
        print(f"Documento Original: '{texto_original[:250]}...'")
        print(f"Tópico Inicial Gerado: {topico_inicial}")
        print(f"Tópico Refinado com RAG: {topico_atualizado}")
        print(f"{'='*58}")

        # (NOVO) Salvar resultados na lista
        resultados_finais.append({
            "documento": texto_original,
            "classificacao_1": topico_inicial,
            "classificacao_2": topico_atualizado
        })
        # (FIM DO NOVO)

    print("\n--- CALCULANDO MÉTRICAS GLOBAIS E SALVANDO CSV ---")

    if not resultados_finais:
        print("Nenhum resultado foi processado. Encerrando.")
        return

    # 1. Agrupar documentos por tópico (usando a classificação 2, refinada)
    topic_docs_map = defaultdict(list)
    documentos_processados = []
    
    for res in resultados_finais:
        # Ignora erros do LLM na análise
        if "ERRO" not in res["classificacao_2"]:
            topic_docs_map[res["classificacao_2"]].append(res["documento"])
            documentos_processados.append(res["documento"])

    global_coherence = 0.0
    global_diversity = 0.0

    if not topic_docs_map:
        print("Não há tópicos válidos para calcular métricas (ex: todos foram ERRO).")
    else:
        # 2. Extrair top 10 palavras para cada tópico
        topics_with_words = extract_top_n_words(topic_docs_map, stop_words, top_n=10)

        # 3. Calcular Coerência (C_v)
        # Usamos os documentos que foram efetivamente agrupados como corpus de referência
        global_coherence = calculate_topic_coherence(
            topics_with_words, 
            documentos_processados, 
            stop_words, 
            coherence_type='c_v'
        )

        # 4. Calcular Diversidade
        global_diversity = calculate_topic_diversity(topics_with_words)

    # 5. Criar DataFrame e adicionar métricas globais
    df = pd.DataFrame(resultados_finais)
    df['topic_coherence'] = global_coherence
    df['topic_diversity'] = global_diversity

    # 6. Formatar colunas conforme solicitado e salvar
    df = df.rename(columns={
        'classificacao_1': 'classificação 1',
        'classificacao_2': 'classificação 2'
    })
    
    # Selecionar e ordenar as colunas
    colunas_finais = ['documento', 'classificação 1', 'classificação 2', 'topic_coherence', 'topic_diversity']
    df = df[colunas_finais]
    
    output_filename = "llm_topic_results.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\nResultados salvos com sucesso em '{output_filename}'")
    print("--- EXPERIMENTO CONCLUÍDO ---")

if __name__ == "__main__":
    main()