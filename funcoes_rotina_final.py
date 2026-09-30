import os
import json
import re
import sys
import time
import contextlib
import logging
from datetime import datetime
import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from ollama import Client

# ==============================================================================
# CONTROLE DE SAÍDA
# ==============================================================================
@contextlib.contextmanager
def silenciar_saida_detalhada():
    """
    Silencia temporariamente:

    - prints das bibliotecas;
    - barras de progresso do tqdm;
    - mensagens em stderr;
    - logs INFO e WARNING do Sentence Transformers;
    - logs INFO e WARNING do Transformers;
    - logs internos do TopicGPT.

    Ao final, todas as configurações são restauradas.
    """

    loggers_silenciados = [
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "transformers",
        "topicgpt_python",
        "httpx",
        "httpcore",
        "openai",
    ]

    estados_anteriores = {}

    for nome_logger in loggers_silenciados:
        logger = logging.getLogger(nome_logger)

        estados_anteriores[nome_logger] = {
            "level": logger.level,
            "disabled": logger.disabled,
            "propagate": logger.propagate,
        }

        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        logger.propagate = False

    # Alguns registros podem estar sendo enviados diretamente ao root logger.
    root_logger = logging.getLogger()
    nivel_root_anterior = root_logger.level
    root_logger.setLevel(logging.CRITICAL + 1)

    try:
        with open(os.devnull, "w", encoding="utf-8") as destino_nulo:
            with contextlib.redirect_stdout(destino_nulo), \
                 contextlib.redirect_stderr(destino_nulo):
                yield

    finally:
        # Restaura o logger raiz.
        root_logger.setLevel(nivel_root_anterior)

        # Restaura individualmente os loggers alterados.
        for nome_logger, estado in estados_anteriores.items():
            logger = logging.getLogger(nome_logger)
            logger.setLevel(estado["level"])
            logger.disabled = estado["disabled"]
            logger.propagate = estado["propagate"]

# ==============================================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# ==============================================================================

# 🛑 IMPORTANTE: Mude para False quando for rodar o experimento oficial na madrugada!
MODO_TESTE = False
# Parâmetros oficiais do artigo TopicGPT para salvar no Log
HIPERPARAMETROS = {
    "max_tokens": 300,
    "temperature": 0.0,
    "top_p": 0.0,
    "auto_correction_limit": 10, # Limite teórico aplicado pela biblioteca/prompt
    "min_doc_frequency_bills": 10,
    "min_doc_frequency_wiki": 5,
    "max_words_per_doc": 250 # Truncate manual para caber no contexto do LLM
}

class ConfiguradorOllama:
    @staticmethod
    def setup_ambiente_topicgpt(host="127.0.0.1", port="11434"):
        base_url = f"http://{host}:{port}/v1"
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = "ollama-key-falsa" 
        print(f"Ambiente configurado! topicgpt_python apontará para: {base_url}")

class PreparadorDatasets:
    """ Baixa, processa, fatia e salva os datasets no formato JSONL esperado pelo TopicGPT """
    
    @staticmethod
    def _truncate_text(texto):
        # Truncate para garantir que caia na janela de contexto do Llama/Gemma
        if not isinstance(texto, str): return ""
        palavras = texto.split()
        return " ".join(palavras[:HIPERPARAMETROS["max_words_per_doc"]])

    @staticmethod
    def preparar_bills():
        print("\n[BILLS] Baixando e processando dataset Congressional Bills...")
        dataset = load_dataset("zli12321/Bills")
        df_train = dataset['train'].to_pandas()
        
        # Mapeamento e Truncate
        df_train['text'] = df_train['summary'].apply(PreparadorDatasets._truncate_text)
        df_train['label_name'] = df_train['topic']
        
        n_gen = 10 if MODO_TESTE else 1000
        n_ass = 10 if MODO_TESTE else 15242
        
        # Amostragem (Garantindo que Atribuição não pegue docs da Geração)
        df_gen = df_train.sample(n=n_gen, random_state=42)
        df_ass = df_train.drop(df_gen.index).sample(n=n_ass, random_state=42)
        
        os.makedirs("datasets/experimento", exist_ok=True)
        path_gen = "datasets/experimento/bills_gen.jsonl"
        path_ass = "datasets/experimento/bills_ass.jsonl"
        
        df_gen[['text', 'label_name']].to_json(path_gen, orient='records', lines=True, force_ascii=False)
        df_ass[['text', 'label_name']].to_json(path_ass, orient='records', lines=True, force_ascii=False)
        
        print(f"[BILLS] Prontos! Geração: {len(df_gen)} | Atribuição: {len(df_ass)}")
        return path_gen, path_ass, len(df_gen), len(df_ass)

    @staticmethod
    def preparar_wiki():
        """
        Baixa a base Wiki oficial disponibilizada pelo projeto TopicGPT,
        preserva o ground truth e cria as amostras de geração e atribuição.

        Fonte oficial:
        https://drive.google.com/drive/folders/1rCTR5ZQQ7bZQoewFA8eqV6glP6zhY31e
        """
        from pathlib import Path

        print("\n[WIKI] Obtendo a base Wiki oficial do TopicGPT com labels...")

        pasta_download = Path("datasets/topicgpt_oficial")
        pasta_download.mkdir(parents=True, exist_ok=True)
        marcador = pasta_download / ".download_concluido"

        if not marcador.exists():
            try:
                import gdown
            except ImportError as erro:
                raise ImportError(
                    "A rotina da Wiki requer o pacote gdown. Instale uma vez com: "
                    "%pip install -U gdown"
                ) from erro

            url_pasta = (
                "https://drive.google.com/drive/folders/"
                "1rCTR5ZQQ7bZQoewFA8eqV6glP6zhY31e?usp=sharing"
            )
            arquivos_baixados = gdown.download_folder(
                url=url_pasta,
                output=str(pasta_download),
                quiet=False,
                use_cookies=False,
            )
            if not arquivos_baixados:
                raise RuntimeError(
                    "Nenhum arquivo foi baixado da pasta oficial do TopicGPT. "
                    "Verifique o acesso da VM à internet e ao Google Drive."
                )
            marcador.write_text("ok", encoding="utf-8")
        else:
            print("[WIKI] Download oficial já existe. Reutilizando arquivos locais.")

        candidatos = sorted(
            p for p in pasta_download.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".json", ".jsonl"}
            and "wiki" in p.name.lower()
        )
        if not candidatos:
            raise FileNotFoundError(
                "O download terminou, mas nenhum JSON/JSONL da Wiki foi localizado "
                f"em '{pasta_download}'."
            )

        frames_validos = []
        relatorio = []
        for caminho in candidatos:
            try:
                try:
                    df = pd.read_json(caminho, lines=True)
                except ValueError:
                    df = pd.read_json(caminho)
            except Exception as erro:
                relatorio.append(f"{caminho}: leitura falhou ({erro})")
                continue

            col_texto = next(
                (c for c in ["text", "document", "content"] if c in df.columns),
                None,
            )
            col_label = next(
                (
                    c for c in [
                        "label_name", "label", "high_level_label",
                        "label_high", "topic", "category"
                    ]
                    if c in df.columns
                ),
                None,
            )
            relatorio.append(
                f"{caminho}: linhas={len(df)}, texto={col_texto}, label={col_label}"
            )
            if col_texto and col_label:
                parte = df[[col_texto, col_label]].copy()
                parte.columns = ["text", "label_name"]
                frames_validos.append(parte)

        if not frames_validos:
            raise ValueError(
                "Nenhum arquivo Wiki baixado contém simultaneamente texto e label.\n"
                + "\n".join(relatorio)
            )

        df_wiki = pd.concat(frames_validos, ignore_index=True)
        df_wiki["text"] = df_wiki["text"].astype(str).str.strip()
        df_wiki["label_name"] = df_wiki["label_name"].apply(
            lambda valor: valor[0] if isinstance(valor, (list, tuple)) and valor else valor
        )
        df_wiki["label_name"] = df_wiki["label_name"].astype(str).str.strip()
        df_wiki = df_wiki[
            df_wiki["text"].ne("")
            & df_wiki["label_name"].ne("")
            & ~df_wiki["label_name"].str.lower().isin({"nan", "none", "n/a"})
        ].copy()
        df_wiki = df_wiki.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df_wiki["text"] = df_wiki["text"].apply(PreparadorDatasets._truncate_text)

        qtd_labels = df_wiki["label_name"].nunique()
        if qtd_labels <= 1:
            raise ValueError(
                f"A base Wiki localizada possui somente {qtd_labels} label distinta. "
                "O ground truth não foi carregado corretamente."
            )

        n_gen = 10 if MODO_TESTE else 1100
        n_ass = 10 if MODO_TESTE else 8024
        necessario = n_gen + n_ass
        if len(df_wiki) < necessario:
            raise ValueError(
                f"A base Wiki possui {len(df_wiki)} documentos válidos, mas são "
                f"necessários {necessario} ({n_gen} + {n_ass})."
            )

        df_gen = df_wiki.sample(n=n_gen, random_state=42)
        df_ass = df_wiki.drop(df_gen.index).sample(n=n_ass, random_state=42)

        os.makedirs("datasets/experimento", exist_ok=True)
        path_gen = "datasets/experimento/wiki_gen.jsonl"
        path_ass = "datasets/experimento/wiki_ass.jsonl"
        df_gen[["text", "label_name"]].to_json(
            path_gen, orient="records", lines=True, force_ascii=False
        )
        df_ass[["text", "label_name"]].to_json(
            path_ass, orient="records", lines=True, force_ascii=False
        )

        print(
            f"[WIKI] Base correta pronta! Documentos válidos: {len(df_wiki)} | "
            f"Labels: {qtd_labels} | Geração: {len(df_gen)} | "
            f"Atribuição: {len(df_ass)}"
        )
        print("[WIKI] Arquivos oficiais identificados:")
        for item in relatorio:
            print(f"  - {item}")

        return path_gen, path_ass, len(df_gen), len(df_ass)

class PipelineMestrado:
    def __init__(self, model_name="gemma2:9b"):
        self.api = "openai" 
        self.model_name = model_name

    def _extrair_topicos_limpos(self, topic_file):
        """Cria o TXT de tópicos no formato aceito pela fase 3."""
        if not os.path.exists(topic_file):
            raise FileNotFoundError(f"Arquivo de tópicos não encontrado: '{topic_file}'")
        if os.path.getsize(topic_file) == 0:
            raise ValueError(f"O arquivo de tópicos está vazio: '{topic_file}'")

        clean_file = topic_file.replace(".json", "_limpo.txt")
        topicos_globais = {}
        regex_topico = re.compile(r"\[(\d+)\]\s*([^:]+):\s*(.*)")

        with open(topic_file, "r", encoding="utf-8") as arquivo:
            for linha in arquivo:
                if not linha.strip():
                    continue
                try:
                    dado = json.loads(linha)
                except json.JSONDecodeError:
                    continue

                resposta = dado.get("refined_responses", dado.get("responses", ""))
                if isinstance(resposta, list):
                    resposta = "\n".join(str(item) for item in resposta)
                if not isinstance(resposta, str) or not resposta.strip():
                    continue

                for linha_resposta in resposta.splitlines():
                    match = regex_topico.search(linha_resposta)
                    if not match:
                        continue
                    nivel = match.group(1).strip()
                    nome = re.sub(r"\s*\(\d+\)\s*$", "", match.group(2)).strip()
                    descricao = match.group(3).strip() or "Descrição não informada"
                    if nome:
                        chave = f"[{nivel}] {nome}"
                        topicos_globais.setdefault(
                            chave,
                            {"lvl": nivel, "nome": nome, "desc": descricao}
                        )

        if not topicos_globais:
            raise ValueError(
                f"Nenhum tópico válido foi extraído de '{topic_file}'. "
                "Verifique 'refined_responses' e 'responses' da fase 2."
            )

        with open(clean_file, "w", encoding="utf-8") as arquivo:
            for info in topicos_globais.values():
                arquivo.write(f"[{info['lvl']}] {info['nome']} (Count: 0): {info['desc']}\n")

        if os.path.getsize(clean_file) == 0:
            raise ValueError(f"O arquivo limpo foi criado vazio: '{clean_file}'")

        # Valida exatamente o padrão exigido por TopicTree.from_topic_list.
        padrao_topicgpt = re.compile(
            r"^\[(\d+)\] (.+) \(Count: (\d+)\)\s?:(.*)$"
        )
        with open(clean_file, "r", encoding="utf-8") as arquivo:
            for numero_linha, linha in enumerate(arquivo, start=1):
                conteudo = linha.strip()
                if conteudo and not padrao_topicgpt.match(conteudo):
                    raise ValueError(
                        f"Linha {numero_linha} incompatível com TopicGPT: {conteudo!r}"
                    )

        return clean_file, len(topicos_globais)

    def gerar_topicos(self, data_file, prompt_file, seed_file, out_file, topic_file):
        from topicgpt_python import generate_topic_lvl1
        print(" -> Iniciando Fase 1: Geração de Tópicos...")
        inicio = time.time()
        
        with silenciar_saida_detalhada():
            generate_topic_lvl1(
                api=self.api,
                model=self.model_name,
                data=data_file, 
                prompt_file=prompt_file,
                seed_file=seed_file,
                out_file=out_file,
                topic_file=topic_file,
                verbose=False  # Silenciado
            )
        
        tempo = time.time() - inicio
        # Conta tópicos gerados para o log
        try:
            with open(topic_file, 'r', encoding='utf-8') as f:
                qtd_topicos = len(f.readlines())
        except: qtd_topicos = 0
        
        print(f"    [✔] Concluído em {tempo:.2f}s | Tópicos encontrados: {qtd_topicos}")
        return qtd_topicos

    def refinar_topicos(
        self,
        prompt_file,
        generation_file,
        topic_file,
        out_file,
        updated_file,
        mapping_file
    ):

        print(" -> Iniciando Fase 2: Refinamento de Tópicos...")
        inicio = time.time()

        with silenciar_saida_detalhada():
            from topicgpt_python import refine_topics

            refine_topics(
                api=self.api,
                model=self.model_name,
                prompt_file=prompt_file,
                generation_file=generation_file,
                topic_file=topic_file,
                out_file=out_file,
                updated_file=updated_file,
                verbose=False,
                remove=True,
                mapping_file=mapping_file
            )

        _, qtd_refinados = self._extrair_topicos_limpos(updated_file)

        tempo = time.time() - inicio

        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos finais consolidados: {qtd_refinados}"
        )

        return qtd_refinados

    def atribuir_topicos(self, data_file, prompt_file, out_file, topic_file):
        print(" -> Iniciando Fase 3: Atribuição de Tópicos...")
        inicio = time.time()

        for descricao, caminho in {
            "dataset de atribuição": data_file,
            "prompt de atribuição": prompt_file,
            "arquivo de tópicos refinados": topic_file,
        }.items():
            if not os.path.exists(caminho):
                raise FileNotFoundError(f"O {descricao} não foi encontrado: '{caminho}'")
            if os.path.getsize(caminho) == 0:
                raise ValueError(f"O {descricao} está vazio: '{caminho}'")

        topic_file_pronto, quantidade_topicos = self._extrair_topicos_limpos(topic_file)

        if os.path.exists(out_file):
            os.remove(out_file)

        try:
            with silenciar_saida_detalhada():
                from topicgpt_python import assign_topics
                assign_topics(
                    api=self.api,
                    model=self.model_name,
                    data=data_file,
                    prompt_file=prompt_file,
                    out_file=out_file,
                    topic_file=topic_file_pronto,
                    verbose=False
                )
        except Exception as erro:
            raise RuntimeError(
                "A fase 3 falhou durante a atribuição de tópicos.\n"
                f"Dataset: {data_file}\n"
                f"Prompt: {prompt_file}\n"
                f"Tópicos originais: {topic_file}\n"
                f"Tópicos limpos: {topic_file_pronto}\n"
                f"Quantidade de tópicos: {quantidade_topicos}\n"
                f"Erro original: {type(erro).__name__}: {erro}"
            ) from erro

        if not os.path.exists(out_file):
            raise FileNotFoundError(f"A fase 3 não criou a saída: '{out_file}'")
        if os.path.getsize(out_file) == 0:
            raise ValueError(f"A fase 3 criou uma saída vazia: '{out_file}'")

        tempo = time.time() - inicio
        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos disponíveis: {quantidade_topicos} | Saída: {out_file}"
        )

class AvaliadorMetricas:
    @staticmethod
    def calcular_purity(y_true, y_pred):
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=0)) / np.sum(matriz)

    @staticmethod
    def calcular_inverse_purity(y_true, y_pred):
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=1)) / np.sum(matriz)

    @staticmethod
    def avaliar_resultados(caminho_arquivo_json, dataset_name):
        y_true = []
        y_pred = []
        labels_originais_set = set()

        with open(caminho_arquivo_json, 'r', encoding='utf-8') as f:
            try: dados = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                dados = [json.loads(linha) for linha in f]

        for item in dados:
            verdadeiro = item.get('label_name')
            predito_bruto = item.get('responses', '') 
            
            if verdadeiro is not None and predito_bruto:
                match = re.search(r"Assignment:\s*\[\d+\]\s*([^:]+)", predito_bruto, re.IGNORECASE)
                if not match: match = re.search(r"\[\d+\]\s*([^:]+)", predito_bruto)
                
                predito_limpo = match.group(1).strip().lower() if match else "indefinido"
                verdadeiro_limpo = str(verdadeiro).strip().lower()

                y_true.append(verdadeiro_limpo)
                y_pred.append(predito_limpo)
                labels_originais_set.add(verdadeiro_limpo)

        # Se for Wiki ou base sem labels classificados, NMI/ARI não podem ser calculados.
        if "N/A".lower() in labels_originais_set or len(labels_originais_set) <= 1:
            print(" -> Aviso: Ground-Truth não detectado (Dataset tipo WIKI). Calculando apenas estatísticas.")
            return {
                "NMI": None, "ARI": None, "HMP": None, "Purity": None, "Inverse_Purity": None,
                "Amostras_Validadas": len(y_true), "Topicos_Criados": len(set(y_pred)), "Topicos_Ground_Truth": 0
            }

        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        purity = AvaliadorMetricas.calcular_purity(y_true, y_pred)
        inv_purity = AvaliadorMetricas.calcular_inverse_purity(y_true, y_pred)
        hmp = 0 if (purity + inv_purity) == 0 else 2 * (purity * inv_purity) / (purity + inv_purity)

        print(f"\n--- Resultados da Avaliação: {dataset_name} ---")
        print(f"NMI: {nmi:.4f} | ARI: {ari:.4f} | HMP: {hmp:.4f}")
        
        return {
            "NMI": nmi, "ARI": ari, "HMP": hmp, "Purity": purity, "Inverse_Purity": inv_purity,
            "Amostras_Validadas": len(y_true), "Topicos_Criados": len(set(y_pred)), "Topicos_Ground_Truth": len(labels_originais_set)
        }

def registrar_log(dataset, metricas, n_gen, n_ass):
    log_file = "experimento_metricas_geral.json"
    registro = {
        "dataset": dataset,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modo_teste": MODO_TESTE,
        "amostra_geracao": n_gen,
        "amostra_atribuicao": n_ass,
        "parametros_llm": HIPERPARAMETROS,
        "metricas": metricas
    }
    
    historico = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            historico = json.load(f)
            
    historico.append(registro)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(historico, f, indent=4, ensure_ascii=False)
    print(f"\n[💾] Log salvo com sucesso em '{log_file}' (Fora da pasta resultados).")

# ==============================================================================
# FLUXO DE EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print(f"=== INICIANDO EXPERIMENTO TOPICGPT {'(MODO TESTE)' if MODO_TESTE else '(MODO COMPLETO)'} ===")
    
    os.makedirs("resultados", exist_ok=True)
    ConfiguradorOllama.setup_ambiente_topicgpt(host="127.0.0.1", port="11434")
    pipeline = PipelineMestrado(model_name="gemma2:9b") # Altere o modelo aqui se necessário

    # 1. PROCESSAR BILLS
    print("\n" + "="*50 + "\nDATASET 1: CONGRESSIONAL BILLS\n" + "="*50)
    file_bills_gen, file_bills_ass, n_gen_b, n_ass_b = PreparadorDatasets.preparar_bills()
    
    qtd_gen_bills = pipeline.gerar_topicos(file_bills_gen, "prompts/generation_1.txt", "prompts/seed_1.md", "resultados/b_geracao.json", "resultados/b_topicos.json")
    qtd_ref_bills = pipeline.refinar_topicos("prompts/refinement.txt", "resultados/b_geracao.json", "resultados/b_topicos.json", "resultados/b_refinamento.json", "resultados/b_top_refinados.json", "resultados/b_map.json")
    pipeline.atribuir_topicos(file_bills_ass, "prompts/assignment.txt", "resultados/b_atribuida.json", "resultados/b_top_refinados.json")
    
    metricas_bills = AvaliadorMetricas.avaliar_resultados("resultados/b_atribuida.json", "BILLS")
    registrar_log("BILLS", metricas_bills, n_gen_b, n_ass_b)

    # 2. PROCESSAR WIKITEXT
    print("\n" + "="*50 + "\nDATASET 2: WIKITEXT-103\n" + "="*50)
    file_wiki_gen, file_wiki_ass, n_gen_w, n_ass_w = PreparadorDatasets.preparar_wiki()
    
    qtd_gen_wiki = pipeline.gerar_topicos(file_wiki_gen, "prompts/generation_1.txt", "prompts/seed_1.md", "resultados/w_geracao.json", "resultados/w_topicos.json")
    qtd_ref_wiki = pipeline.refinar_topicos("prompts/refinement.txt", "resultados/w_geracao.json", "resultados/w_topicos.json", "resultados/w_refinamento.json", "resultados/w_top_refinados.json", "resultados/w_map.json")
    pipeline.atribuir_topicos(file_wiki_ass, "prompts/assignment.txt", "resultados/w_atribuida.json", "resultados/w_top_refinados.json")
    
    metricas_wiki = AvaliadorMetricas.avaliar_resultados("resultados/w_atribuida.json", "WIKITEXT")
    registrar_log("WIKITEXT", metricas_wiki, n_gen_w, n_ass_w)
    
    print("\n🎉 === EXPERIMENTO FINALIZADO COM SUCESSO! === 🎉")