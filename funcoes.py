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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment # Necessário para o HMP

# Para o pré-processamento de texto (mantido caso precise de alguma utilidade do gensim)
import gensim

# Importa a classe TopicManager do arquivo topic_manager.py
# (O TopicManager precisa de 're' e 'collections.defaultdict' que já estão acima)
from topic_manager import TopicManager 


# --- Constantes de Configuração ---
# OLLAMA_HOST = "http://164.41.75.221:11434"  # Host Ollama, conforme fornecido
OLLAMA_HOST = '127.0.0.1:11434'  # Host Ollama, conforme fornecido
LLM_MODEL = "llama3.1" # Modelo LLM a ser usado no experimento, conforme solicitado
DATASET_NAME = "cardiffnlp/tweet_topic_single"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Modelo de embedding eficiente para a tarefa
NUM_RANDOM_SAMPLES_FOR_LLM = 1000  # Quantidade de documentos para gerar tópicos com o LLM
TOP_K_SIMILAR = 3 # Número de documentos similares a serem recuperados para o contexto
# topicos_existentes_str: REMOVIDA. Agora a lista é dinâmica e passada como argumento.


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

    def _chamar_llm(self, prompt: str) -> str:
        """
        Função auxiliar para abstrair a chamada real ao LLM.
        """
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


    def _criar_prompt_geracao_topico(self, texto_documento: str, topicos_historicos_str: str) -> str:
        """
        Helper para criar o prompt formatado para a Geração/Assimilação Incremental de Tópicos (Etapa 1).
        """

        prompt = f"""
        You will receive a document and a list of previously identified top-level topics. Your task is to identify the **SINGLE MOST** generalizable top-level topic mentioned in the document.

		The output must be a single, short topic label that can act as a top-level topic.

		[CURRENT TOPIC LIST (HISTORY)]
		{topicos_historicos_str}

		[Examples]

		Example 1: Document about agricultural policies
		Document:
		Saving Essential American Sailors Act or SEAS Act - Amends the Moving Ahead for Progress in the 21st Century Act (MAP-21)
		to repeal the Act’s repeal of the agricultural export requirements that: (1) 25 of the gross tonnage of certain agricultural
		commodities or their products exported each fiscal year be transported on U.S. commercial vessels...
		Your response:
		Agriculture

		Example 2: Document about duties suspension
		Document:
		Amends the Harmonized Tariff Schedule of the United States to suspend temporarily the duty on mixtures containing Fluopyram.
		Your response:
		Trade

		[STRICT INSTRUCTIONS]

		1. **HISTORY CHECK:** Analyze the **[CURRENT TOPIC LIST (HISTORY)]**. If the most generalizable topic for the current document is a **paraphrase or equivalent** of an existing topic in the list, you **MUST use the existing label** from the list for consistency.
		2. **Identify the main, most generalizable topic** in the document. The topic must be broad enough to accommodate future subtopics.
		3. **Your output must be the topic label ONLY**. Do not include the level indicator, a description, or any explanatory text.
		4. If the document contains **NO identifiable top-level topic**, return "None".
		5. If multiple topics are found, choose the single most encompassing one.

		Document to classify:
		"{texto_documento}"

		Respond ONLY with the topic label or "None". Do not output anything else (no code, no descriptions, no introductory phrases).

		Topic Label:
        """

        return prompt.strip()
    
    def gerar_topico_incremental(self, texto_documento: str, topicos_str: str) -> str:
        """
        Etapa 1: Gera/Assimila o tópico inicial, usando a lista histórica de tópicos.
        O LLM deve retornar APENAS o rótulo do tópico ou 'None'.
        """
        prompt = self._criar_prompt_geracao_topico(texto_documento, topicos_str)
        print(f"\nEnviando documento para o LLM:\n'{texto_documento[:100]}...'")
        
        topico = self._chamar_llm(prompt) 
        
        print(f"LLM respondeu com o tópico inicial: '{topico}'")
        # Limpeza da resposta para garantir apenas o rótulo
        return topico

    def consolidar_topicos(self, lista_de_topicos_formatada: str) -> str:
        """
        Etapa de Consolidação: Pede ao LLM para fundir tópicos duplicados/paráfrases.
        O LLM deve retornar a string de modificação ou 'None'.
        """
        prompt = self._criar_prompt_consolidacao_topicos(lista_de_topicos_formatada)
        
        # O retorno é a string bruta que o TopicManager irá parsear
        return self._chamar_llm(prompt)
    
    def _criar_prompt_consolidacao_topicos(self, lista_de_topicos_formatada: str) -> str:
        """
        Cria o prompt para a etapa de consolidação (fusão de duplicados) dos tópicos.
        (Corrigido para usar o formato complexo de saída)
        -   An updated description that encompasses the meaning of the merged topics.
        -   Example: [1] Employment Taxes: Mentions taxation report and requirement for employer ([1] Employer Taxes, [1] Employment Tax Reporting)

        """
        prompt = f"""
        You will receive a list of top-level topics that have been generated incrementally. Your task is to perform **topic consolidation** by merging topics that are **semantically equivalent, paraphrases, or near duplicates** of one another.

        This process aims to reduce redundancy and ensure a consistent set of top-level categories.

        [STRICT RULES]

        1.  **MERGING:** Merge only topics that represent the **same underlying concept** (i.e., they are redundant).
        2.  **OUTPUT FORMAT:** When merging, the output must be a single line containing:
            -   A level indicator (e.g., "[1]").
            -   The new, consolidated label (must be the most appropriate and generalizable term).
            -   The original topics merged, listed in parentheses (e.g., ([1] Original Topic A, [1] Original Topic B)).
        3.  **NO MERGE:** If no topics require merging, return the string **"None"**.
        4.  **CONCISENESS:** Output ONLY the merged line(s) or the word "None". Do not output introductory text, rules, or the full topic list again.

        [Examples]

        Example 1: Merging paraphrases
        Topic List:
        [1] Employer Taxes
        [1] Employment Tax Reporting
        Your response:
        [1] Employment Taxes

        Example 2: Merging more general concepts
        Topic List:
        [2] Statistics
        [2] Digital Literacy
        [2] Telecommunications
        Your response:
        [2] Technology

        [Topic List to Consolidate]
        {lista_de_topicos_formatada}

        Output the modification or "None" where appropriate.
        Your response:
        """
        return prompt.strip()

    def actualizar_topico_com_contexto(self, topico_inicial: str, documentos_contexto: List[str]) -> str:
        """Envia o tópico inicial e um contexto de documentos similares para o LLM refinar o tópico."""
        prompt = self._criar_prompt_actualizacao_topico(topico_inicial, documentos_contexto)
        try:
            print(f"Enviando contexto para refinar o tópico '{topico_inicial}'...")
            topico_actualizado = self._chamar_llm(prompt)
            print(f"LLM refinou para o tópico: '{topico_actualizado}'")
            return topico_actualizado
        except Exception as e:
            print(f"Erro ao contatar o LLM para refinamento: {e}")
            return "ERRO_REFINAMENTO_LLM"
    
    def _criar_prompt_actualizacao_topico(self, topico_inicial: str, documentos_contexto: List[str]) -> str:
        """Helper para criar o prompt de refinamento de tópico com base em contexto (Etapa 2)."""
        contexto_str = "\n\n".join([f"Documento similar {i+1}:\n\"{doc}\"" for i, doc in enumerate(documentos_contexto)])

        prompt = f"""
        You are currently in the topic classification refinement stage.
	    The initial proposed topic for a document was: "{topico_inicial}".

	    Below are semantically similar documents retrieved from the vector store.
	    Analyze this additional context and the initial topic.
	    Your objective is to **reclassify** the document by choosing the single most appropriate topic label based on the semantic consensus of the provided context.

	    Context from similar documents (Vector Store):
	    {contexto_str}

	    [STRICT RULES FOR OUTPUT]

	    1. **MANDATORY FORMAT:** The output must be **EXACTLY ONE** word or a short compound name (e.g., "Employment Taxes", "Brexit").
	    2. **NO EXPLANATIONS:** You **MUST NOT** include any rationale, description, explanation, or introductory phrases (like "Refined Topic is:", "Explanation:", or any discussion about why you chose the topic).
	    3. **CONSISTENCY:** The refined topic label should be the most **consistent** and **appropriate** term to categorize the document based on the semantic consensus of the context.
	    4. **AFFIRMATION:** If the initial proposed topic is the most appropriate and consistent label based on the context, you must output the initial topic label.

	    [Examples]

	    Example 1: Initial Topic Refinement based on Context
	    Initial Proposed Topic: "Immigration"
	    Context from similar documents:
	    - Document A: discusses new visa policies and requirements for highly skilled workers.
	    - Document B: describes the legal process for obtaining permanent residency.
	    - Document C: compares different country's asylum seeking processes.
	    Refined Topic:
	    Immigration

	    Example 2: Initial Topic Rejection and Reclassification
	    Initial Proposed Topic: "Entertainment"
	    Context from similar documents:
	    - Document A: details the latest box office numbers for summer action movies.
	    - Document B: provides a review of a new television series airing on a streaming platform.
	    - Document C: reports on the annual awards ceremony for a film industry.
	    Refined Topic:
	    Celebrities

	    Example 3: Reclassification to a more specific topic
	    Initial Proposed Topic: "Technology"
	    Context from similar documents:
	    - Document A: Explains the function and components of a new 5G network standard.
	    - Document B: Discusses regulations related to wireless service providers and consumer rights in internet access.
	    - Document C: Mentions government policies regarding the rollout of fiber optic infrastructure.
	    Refined Topic:
	    Telecommunications

	    Based on the context, what is the single most consistent and appropriate topic for the document?

	    **[FINAL RESPONSE MUST BE THE TOPIC LABEL ONLY]** """

        return prompt.strip()


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
    
# --- Funções de Avaliação de Tópicos (Mantidas mas não usadas nas métricas finais) ---

def preprocess_text(text: str, stop_words: set) -> List[str]:
    # ... (código existente) ...
    """
    Tokeniza, remove stopwords, pontuação e palavras curtas.
    """
    # simple_preprocess faz a tokenização e passa para minúsculo
    return [word for word in simple_preprocess(text) if word not in stop_words and len(word) > 2]

def extract_top_n_words(topic_docs: Dict[str, List[str]], stop_words: set, top_n: int = 10) -> Dict[str, List[str]]:
    # ... (código existente) ...
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
    # ... (código existente) ...
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
    # ... (código existente) ...
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


# --- Funções de Métricas de Agrupamento ---

def purity_score(y_true, y_pred):
    """Calcula a Purity Score."""
    # Mapeamento dos rótulos (cruzamento entre GT e Preditos)
    contingency_matrix = pd.crosstab(y_true, y_pred)
    
    # Encontra o maior valor em cada coluna (máxima concordância por cluster predito)
    purity = np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)
    return purity

def inverse_purity_score(y_true, y_pred):
    """Calcula a Inverse Purity (também chamada de Homogeneidade em algumas definições)."""
    # É a Purity Score calculada com os rótulos trocados
    return purity_score(y_pred, y_true)

def calculate_harmonic_mean_purity(y_true, y_pred):
    """Calcula o Harmonic Mean Purity (HMP) a partir da Purity e Inverse Purity."""
    purity = purity_score(y_true, y_pred)
    inverse_purity = inverse_purity_score(y_true, y_pred)

    if purity + inverse_purity == 0:
        return 0.0
    
    # HMP = 2 * (Purity * Inverse Purity) / (Purity + Inverse Purity)
    hmp = 2 * purity * inverse_purity / (purity + inverse_purity)
    return hmp


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

    # Download de recursos NLTK (stopwords)
    print("Baixando stopwords do NLTK...")
    try:
        import nltk
        from nltk.corpus import stopwords
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    except ImportError:
        # Se nltk não estiver instalado
        print("Aviso: NLTK não encontrado. Instale 'nltk' para usar stopwords.")
        stop_words = set()
    else:
        stop_words = set(stopwords.words('english'))

    # 2. Inicialização dos componentes
    analisador_llm = AnalisadorLLM(host=OLLAMA_HOST, model=LLM_MODEL)
    vector_store = VectorStore(model_name=EMBEDDING_MODEL_NAME)
    
    # NOVO: Inicializa o gerenciador de tópicos
    topic_manager = TopicManager(analisador_llm)
    
    # 3. Construção do Vector Store da base completa
    corpus_embeddings = vector_store.create_embeddings(documentos, show_progress=True)
    vector_store.build_index(corpus_embeddings)
    
    # 4. Análise com LLM e RAG (em uma amostra)
    print(f"\n--- INICIANDO PROCESSO DE CLASSIFICAÇÃO PARA {NUM_RANDOM_SAMPLES_FOR_LLM} AMOSTRAS ---")
    amostra_docs = dataset.shuffle(seed=42).select(range(NUM_RANDOM_SAMPLES_FOR_LLM))

    resultados_finais = [] # Lista para salvar os resultados brutos
    
    for i, doc in enumerate(amostra_docs):
        print(f"\n{'='*20} Processando Amostra {i+1}/{NUM_RANDOM_SAMPLES_FOR_LLM} {'='*20}")
        texto_original = doc['text']
        
        # Etapa 1: Gerar/Assimilar tópico inicial (Classificação Incremental)
        topicos_historicos_str = topic_manager.get_topicos_str_para_geracao()
        # O método de prompt AGORA usa topicos_historicos_str
        topico_inicial = analisador_llm.gerar_topico_incremental(texto_original, topicos_historicos_str)
        
        # Limpeza básica do tópico inicial
        topico_inicial = topico_inicial.split('\n')[0].strip()
        
        if "ERRO" in topico_inicial or topico_inicial == "None":
            topico_atualizado = topico_inicial 
            print(f"Documento pulado. Tópico inicial: {topico_inicial}")
            
        else:
            # Adiciona o tópico (ou confirma se já existe) na lista de tópicos únicos
            topic_manager.add_topic(topico_inicial) 
            
            # Etapa 2, 3 e 4: RAG e Refinamento
            query_text = f"Tópico: {topico_inicial}. Documento: {texto_original}"
            query_embedding = vector_store.create_embeddings([query_text], show_progress=False)
            
            print(f"Buscando {TOP_K_SIMILAR} documentos similares para a consulta...")
            _, similar_indices = vector_store.search(query_embedding, k=TOP_K_SIMILAR)
            
            documentos_contexto = [documentos[idx] for idx in similar_indices[0]]
            
            topico_atualizado = analisador_llm.actualizar_topico_com_contexto(topico_inicial, documentos_contexto)

            # Exibir resultados consolidados para esta amostra
            print("\n--- RESULTADO DA AMOSTRA ---")
            #print(f"Documento Original: '{texto_original[:250]}...'")
            print(f"Tópico Inicial Gerado: {topico_inicial}")
            print(f"Tópico Refinado com RAG: {topico_atualizado}")
            print(f"{'='*58}")

        # Salvar resultados brutos (pré-consolidação)
        resultados_finais.append({
            "documento": texto_original,
            "classificacao_1": topico_inicial,
            "classificacao_2": topico_atualizado,
            "ground_truth_topic": doc.get('label', 'N/A') # Adicionando o ground truth (se o dataset tiver a coluna 'label')
        })
    
    # FIM DO LOOP DE CLASSIFICAÇÃO
    
    # 5. NOVA ETAPA: CONSOLIDAÇÃO DE TÓPICOS EM LOTE (Uma única vez)
    print(f"\n\n*** EXECUTANDO CONSOLIDAÇÃO DE TÓPICOS em {len(topic_manager.topicos_unicos)} rótulos únicos... ***")
    
    mapeamento_consolidado = topic_manager.consolidar_topicos()
    
    print(f"Tópicos únicos após consolidação: {list(topic_manager.topicos_unicos)}")
    
    # 6. APLICAR MAPEAMENTO AOS RESULTADOS FINAIS
    print("\nAPLICANDO mapeamento de consolidação aos resultados...")
    for res in resultados_finais:
        # APLICAR APENAS NA CLASSIFICAÇÃO 2 (Refinada), pois a Classificação 1 é o tópico incremental "bruto"
        topico_refinado = res["classificacao_2"]
        if topico_refinado in mapeamento_consolidado:
             res["classificacao_2_final"] = mapeamento_consolidado[topico_refinado]
        else:
             # Mantém o tópico se ele não foi modificado (e.g., "None" ou "ERRO")
             res["classificacao_2_final"] = topico_refinado 

    # 7. CÁLCULO DE MÉTRICAS E SALVAMENTO 
    
    print("\n--- CALCULANDO MÉTRICAS GLOBAIS E SALVANDO CSV ---")

    if not resultados_finais:
        print("Nenhum resultado foi processado. Encerrando.")
        return

    # 1. Agrupar rótulos (Ground Truth e Preditos) para documentos válidos
    y_true = []
    y_pred = []
    documentos_validos = []

    for res in resultados_finais:
        # Usa a coluna final após a consolidação
        if "ERRO" not in res["classificacao_2_final"] and res["ground_truth_topic"] != 'N/A':
            y_true.append(res["ground_truth_topic"])
            y_pred.append(res["classificacao_2_final"]) # **CORRIGIDO: USANDO classificacao_2_final**
            documentos_validos.append(res)
        elif res["ground_truth_topic"] == 'N/A':
            print("AVISO: Chave 'ground_truth_topic' não encontrada em um ou mais resultados. Ignorando para métricas.")

    global_hmp = 0.0
    global_nmi = 0.0
    global_ari = 0.0

    if not y_true:
        print("Não há documentos válidos (sem erro ou sem ground truth) para calcular métricas de agrupamento.")
    else:
        # 2. Calcular Harmonic Mean Purity (HMP)
        global_hmp = calculate_harmonic_mean_purity(y_true, y_pred)
        
        # 3. Calcular Normalized Mutual Information (NMI)
        global_nmi = normalized_mutual_info_score(y_true, y_pred)
        
        # 4. Calcular Adjusted Rand Index (ARI)
        global_ari = adjusted_rand_score(y_true, y_pred)
        
        print(f"\nResultados das Métricas:")
        print(f" - Harmonic Mean Purity (HMP): {global_hmp:.4f}")
        print(f" - Normalized Mutual Information (NMI): {global_nmi:.4f}")
        print(f" - Adjusted Rand Index (ARI): {global_ari:.4f}")


    # 5. Criar DataFrame e adicionar métricas globais
    # Usamos resultados_finais pois queremos todas as colunas (incluindo as de erro)
    df = pd.DataFrame(resultados_finais) 

    # Adicionar as métricas globais ao DataFrame completo (se houver dados válidos)
    if not df.empty and y_true:
        df['harmonic_mean_purity'] = global_hmp
        df['normalized_mutual_information'] = global_nmi
        df['adjusted_rand_index'] = global_ari
        
        # 6. Formatar colunas conforme solicitado e salvar
        df = df.rename(columns={
            'classificacao_1': 'classificação 1 (Incremental Bruto)',
            'classificacao_2': 'classificação 2 (Refinada Bruta)',
            'classificacao_2_final': 'classificação 2 (Refinada e Consolidada)', # Coluna usada para métricas
            'ground_truth_topic': 'Ground Truth'
        })
        
        # Selecionar e ordenar as colunas
        colunas_finais = ['documento', 'Ground Truth', 'classificação 1 (Incremental Bruto)', 
                          'classificação 2 (Refinada e Consolidada)', 'harmonic_mean_purity', 
                          'normalized_mutual_information', 'adjusted_rand_index']
        
        df = df.reindex(columns=colunas_finais)    
        output_filename = "llm_topic_results.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print(f"\nResultados salvos com sucesso em '{output_filename}'")
        print("--- EXPERIMENTO CONCLUÍDO ---")
    elif df.empty:
        print("\nDataFrame de resultados vazio. Nada a salvar.")


if __name__ == "__main__":
    main()