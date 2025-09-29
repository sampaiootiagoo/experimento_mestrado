# -*- coding: utf-8 -*-

"""
Esqueleto de código para o experimento de mestrado comparando
geração de tópicos com LLM (Llama) vs. LDA.
"""

# --- Importações de Bibliotecas ---
import os
import re
from typing import List, Dict, Any, Tuple

# Para interagir com o LLM via Ollama
from ollama import Client

# Para carregar o dataset
from datasets import load_dataset

# Para o modelo LDA
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

# Para o pré-processamento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Configurações Iniciais ---
# Baixando recursos necessários do NLTK (executar apenas na primeira vez)
try:
    stopwords.words('english')
except LookupError:
    print("Baixando recursos do NLTK (stopwords, punkt, wordnet)...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# --- Constantes de Configuração ---
OLLAMA_HOST = "'http://164.41.75.221:11434'"  # Mude para o link correto do seu host Ollama
LLM_MODEL = "llama4" # Recomendo usar llama3 que é mais recente que o 4 (que não existe oficialmente)
DATASET_NAME = "cardiffnlp/tweet_topic_single"
NUM_RANDOM_SAMPLES_FOR_LLM = 5  # Quantidade de documentos para gerar tópicos com o LLM
NUM_TOPICS_LDA = 10  # Número de tópicos que o LDA deve encontrar

# --- Definição das Classes ---

class AnalisadorLLM:
    """
    Classe para encapsular a interação com o Large Language Model (LLM) via Ollama.
    """
    def __init__(self, host: str, model: str):
        """
        Inicializa o cliente Ollama.
        :param host: URL do host onde o Ollama está rodando.
        :param model: Nome do modelo a ser utilizado (ex: 'llama3').
        """
        print(f"Inicializando cliente LLM para o host '{host}' e modelo '{model}'...")
        self.client = Client(host=host)
        self.model = model

    def _criar_prompt_geracao_topico(self, texto_documento: str) -> str:
        """Helper para criar o prompt formatado para o LLM."""
        prompt = f"""
        Analise o seguinte documento e gere um único tópico principal que o descreva.
        O tópico deve ser uma ou duas palavras, como "Business", "Technology", "Health", "Sports" ou "Politics".
        Responda apenas com o tópico e nada mais.

        Documento: "{texto_documento}"

        Tópico:
        """
        return prompt.strip()

    def gerar_topico_para_documento(self, texto_documento: str) -> str:
        """
        Envia um documento para o LLM e retorna o tópico gerado.
        
        Este método combina o envio do prompt e o tratamento da resposta.
        
        :param texto_documento: O texto do documento a ser analisado.
        :return: Uma string contendo o tópico identificado pelo LLM.
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
            # Extrai o conteúdo da resposta e remove espaços em branco extras
            topico = response['message']['content'].strip()
            print(f"LLM respondeu com o tópico: '{topico}'")
            return topico
        except Exception as e:
            print(f"Erro ao contatar o LLM: {e}")
            return "ERRO_LLM"


class ProcessadorDeTexto:
    """
    Classe para realizar o pré-processamento e limpeza de textos para análise.
    """
    def __init__(self, lingua: str = 'english'):
        """
        Inicializa o processador, carregando stopwords e o lematizador.
        """
        self.stop_words = set(stopwords.words(lingua))
        self.lemmatizer = WordNetLemmatizer()

    def limpar_e_tokenizar(self, texto: str) -> List[str]:
        """
        Aplica um pipeline de limpeza completo no texto, necessário para o LDA.
        - Converte para minúsculas
        - Remove pontuação e números
        - Tokeniza (divide em palavras)
        - Remove stopwords
        - Lematiza as palavras
        
        :param texto: Texto original.
        :return: Lista de palavras (tokens) limpas.
        """
        # 1. Minúsculas e remoção de caracteres não-alfabéticos
        texto = re.sub(r'[^a-zA-Z\s]', '', texto, re.I|re.A)
        texto = texto.lower()
        
        # 2. Tokenização
        tokens = word_tokenize(texto)
        
        # 3. Remove stopwords e lematiza
        tokens_limpos = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens_limpos


class ModeloLDA:
    """
    Classe para treinar e analisar um modelo Latent Dirichlet Allocation (LDA).
    """
    def __init__(self, num_topicos: int):
        """
        :param num_topicos: O número de tópicos a serem extraídos dos documentos.
        """
        self.num_topicos = num_topicos
        self.modelo_lda = None
        self.dicionario = None
        self.corpus = None

    def treinar(self, documentos: List[List[str]]):
        """
        Treina o modelo LDA com base nos documentos pré-processados.
        
        :param documentos: Uma lista de documentos, onde cada documento é uma lista de tokens.
        """
        print(f"\nTreinando modelo LDA com {self.num_topicos} tópicos...")
        
        # Cria o dicionário e o corpus (formato Bag-of-Words)
        self.dicionario = Dictionary(documentos)
        self.corpus = [self.dicionario.doc2bow(doc) for doc in documentos]
        
        # Treina o modelo
        # Usando LdaMulticore para aproveitar múltiplos processadores
        self.modelo_lda = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dicionario,
            num_topics=self.num_topicos,
            random_state=100,
            chunksize=100,
            passes=10,
            workers=os.cpu_count() - 1 # Usa todos os cores exceto um
        )
        print("Treinamento do LDA concluído.")

    def exibir_topicos(self):
        """Exibe as palavras mais importantes para cada tópico encontrado."""
        if not self.modelo_lda:
            print("O modelo LDA ainda não foi treinado.")
            return
            
        print("\nTópicos encontrados pelo LDA:")
        topicos = self.modelo_lda.print_topics(num_words=5)
        for i, topico in enumerate(topicos):
            print(f"Tópico {i}: {topico[1]}")


# --- Função Principal de Execução do Experimento ---
def main():
    """
    Orquestra a execução do experimento.
    """
    # 1. Carregamento do Dataset
    print(f"Carregando dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split='train_all')
    # Convertendo para uma lista de textos para facilitar a manipulação
    documentos = dataset['text']
    print(f"{len(documentos)} documentos carregados.")

    # 2. Análise com LLM (em uma amostra)
    print("\n--- INICIANDO ANÁLISE COM LLM ---")
    amostra_indices = gensim.utils.simple_preprocess # Usando uma função para pegar índices aleatórios
    amostra_docs = dataset.shuffle(seed=42).select(range(NUM_RANDOM_SAMPLES_FOR_LLM))
    
    analisador_llm = AnalisadorLLM(host=OLLAMA_HOST, model=LLM_MODEL)
    
    # Dicionário para guardar o resultado do LLM (útil para o RAG depois)
    resultados_llm = {}
    for doc in amostra_docs:
        texto = doc['text']
        topico_gerado = analisador_llm.gerar_topico_para_documento(texto)
        resultados_llm[texto] = topico_gerado

    # Aqui você usaria 'resultados_llm' (tópico + documento) para a etapa do RAG
    print("\nEtapa do LLM concluída. Os resultados estão prontos para o RAG.")
    # Exemplo: print(resultados_llm)

    # 3. Análise com LDA (na base completa)
    print("\n--- INICIANDO ANÁLISE COM LDA ---")
    processador = ProcessadorDeTexto()
    
    # Processa todos os documentos para o LDA
    documentos_processados = [processador.limpar_e_tokenizar(doc) for doc in documentos]
    
    # Treina e exibe os tópicos do LDA
    lda = ModeloLDA(num_topicos=NUM_TOPICS_LDA)
    lda.treinar(documentos_processados)
    lda.exibir_topicos()

    print("\n--- EXPERIMENTO CONCLUÍDO ---")


if __name__ == "__main__":
    main()