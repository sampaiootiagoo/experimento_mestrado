import os
import json
import re
import sys
import time
import contextlib
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
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
    """Baixa e prepara os quatro arquivos oficiais usados no experimento."""

    LINKS_GOOGLE_DRIVE = {
        "bills_test": {
            "id": "1sVy7P4XO31brrfmr08jF1SsUP3GMGYTN",
            "nome": "bills_test.metadata.jsonl",
        },
        "bills_train": {
            "id": "1FqctpGokqejEasqJvkDTK1ucM3ZrRZHL",
            "nome": "bills_train.metadata.jsonl",
        },
        "wiki_test": {
            "id": "1yVgiwXedLrtP-AD4FtcymtAL2rXr2Fqs",
            "nome": "wiki_test.metadata.jsonl",
        },
        "wiki_train": {
            "id": "1yJ1ErbDKRifTcliaqF3rl5srvhQl13FC",
            "nome": "wiki_train.metadata.jsonl",
        },
    }

    @staticmethod
    def _truncate_text(texto):
        if not isinstance(texto, str):
            return ""
        palavras = texto.split()
        return " ".join(palavras[:HIPERPARAMETROS["max_words_per_doc"]])

    @staticmethod
    def _baixar_arquivo_drive(chave):
        try:
            import gdown
        except ImportError as erro:
            raise ImportError(
                "Instale o gdown no ambiente atual com: %pip install -U gdown"
            ) from erro

        info = PreparadorDatasets.LINKS_GOOGLE_DRIVE[chave]
        pasta = Path("datasets/topicgpt_oficial")
        pasta.mkdir(parents=True, exist_ok=True)
        destino = pasta / info["nome"]

        if destino.exists() and destino.stat().st_size > 0:
            print(
                f"[DOWNLOAD] Reutilizando {destino} "
                f"({destino.stat().st_size / 1024**2:.1f} MB)."
            )
            return destino

        temporario = destino.with_suffix(destino.suffix + ".part")
        temporario.unlink(missing_ok=True)
        url = f"https://drive.google.com/uc?id={info['id']}"
        print(f"[DOWNLOAD] Baixando {info['nome']}...")

        resultado = gdown.download(
            url=url,
            output=str(temporario),
            quiet=False,
            #fuzzy=True,
            use_cookies=False,
        )
        if not resultado or not temporario.exists() or temporario.stat().st_size == 0:
            temporario.unlink(missing_ok=True)
            raise RuntimeError(f"Falha no download de {info['nome']}.")

        temporario.replace(destino)
        print(
            f"[DOWNLOAD] Concluído: {destino} "
            f"({destino.stat().st_size / 1024**2:.1f} MB)."
        )
        return destino

    @staticmethod
    def _achatar_registro(registro, prefixo=""):
        saida = {}
        for chave, valor in registro.items():
            nome = f"{prefixo}.{chave}" if prefixo else str(chave)
            if isinstance(valor, dict):
                saida.update(PreparadorDatasets._achatar_registro(valor, nome))
            else:
                saida[nome] = valor
        return saida

    @staticmethod
    def _selecionar_coluna(colunas, candidatos):
        mapa = {str(c).lower(): c for c in colunas}
        for candidato in candidatos:
            if candidato.lower() in mapa:
                return mapa[candidato.lower()]
        for candidato in candidatos:
            sufixo = "." + candidato.lower()
            encontrados = [original for baixo, original in mapa.items() if baixo.endswith(sufixo)]
            if encontrados:
                return encontrados[0]
        return None

    @staticmethod
    def _ler_jsonl_com_labels(caminho, dataset_nome):
        registros = []
        colunas_texto = [
            "text", "summary", "document", "content", "article", "body"
        ]
        # Prioriza o nível alto, usado na avaliação principal do TopicGPT.
        colunas_label = [
            "label_high", "high_level_label", "high_label", "level_1_label",
            "label_level_1", "topic", "label_name", "label", "category"
        ]

        with open(caminho, "r", encoding="utf-8") as arquivo:
            for numero_linha, linha in enumerate(arquivo, start=1):
                if not linha.strip():
                    continue
                try:
                    bruto = json.loads(linha)
                except json.JSONDecodeError as erro:
                    raise ValueError(
                        f"JSON inválido em {caminho}, linha {numero_linha}: {erro}"
                    ) from erro

                plano = PreparadorDatasets._achatar_registro(bruto)
                coluna_texto = PreparadorDatasets._selecionar_coluna(
                    plano.keys(), colunas_texto
                )
                coluna_label = PreparadorDatasets._selecionar_coluna(
                    plano.keys(), colunas_label
                )

                if coluna_texto is None or coluna_label is None:
                    if numero_linha == 1:
                        raise ValueError(
                            f"Não foi possível identificar texto e label em {caminho}. "
                            f"Campos encontrados: {sorted(plano.keys())}"
                        )
                    continue

                texto = PreparadorDatasets._truncate_text(plano[coluna_texto])
                label = plano[coluna_label]
                if isinstance(label, (list, tuple)):
                    label = label[0] if label else None
                label = "" if label is None else str(label).strip()

                if texto and label and label.lower() not in {"n/a", "na", "nan", "none", "null"}:
                    registros.append({"text": texto, "label_name": label})

        df = pd.DataFrame(registros).drop_duplicates(subset=["text"]).reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Nenhum registro válido foi lido de {caminho}.")
        qtd_labels = df["label_name"].nunique(dropna=True)
        if qtd_labels <= 1:
            raise ValueError(
                f"{dataset_nome}: somente {qtd_labels} label distinta em {caminho}."
            )
        print(
            f"[{dataset_nome}] {caminho.name}: {len(df)} documentos válidos | "
            f"{qtd_labels} labels."
        )
        return df

    @staticmethod
    def _preparar_dataset(
    prefixo,
    nome_exibicao,
    n_gen,
    n_ass,
    ):
        """
        Baixa e lê as bases de treino e teste, une as observações,
    remove duplicidades e cria amostras independentes para:

        1. geração de tópicos;
    2. atribuição de tópicos.

        Uma observação selecionada para geração nunca será utilizada
        na atribuição.
        """

        caminho_train = (
            PreparadorDatasets._baixar_arquivo_drive(
                f"{prefixo}_train"
            )
        )

        caminho_test = (
            PreparadorDatasets._baixar_arquivo_drive(
                f"{prefixo}_test"
            )
        )

        df_train = (
            PreparadorDatasets._ler_jsonl_com_labels(
                caminho_train,
                nome_exibicao,
            )
        )

        df_test = (
            PreparadorDatasets._ler_jsonl_com_labels(
                caminho_test,
                nome_exibicao,
            )
        )

        quantidade_train = len(df_train)
        quantidade_test = len(df_test)

        print(
            f"[{nome_exibicao}] Registros antes da união | "
            f"Train: {quantidade_train} | "
            f"Test: {quantidade_test}"
        )

        # Acrescenta a origem somente para fins de rastreabilidade.
        df_train = df_train.copy()
        df_test = df_test.copy()

        df_train["origem"] = "train"
        df_test["origem"] = "test"

        # Une treino e teste em uma única base.
        df_total = pd.concat(
            [
                df_train,
                df_test,
            ],
            ignore_index=True,
        )

        quantidade_antes_deduplicacao = len(df_total)

        # Remove documentos repetidos entre treino e teste.
        # A deduplicação considera o conteúdo textual do documento.
        df_total = (
            df_total
            .drop_duplicates(
                subset=["text"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        quantidade_duplicados = (
            quantidade_antes_deduplicacao
            - len(df_total)
        )

        print(
            f"[{nome_exibicao}] Base total após a união: "
            f"{len(df_total)} documentos únicos"
        )

        print(
            f"[{nome_exibicao}] Duplicidades removidas: "
            f"{quantidade_duplicados}"
        )

        quantidade_labels = (
            df_total["label_name"]
            .nunique(dropna=True)
        )

        if quantidade_labels <= 1:
            raise ValueError(
                f"{nome_exibicao}: a base total possui somente "
                f"{quantidade_labels} label distinta."
            )

        print(
            f"[{nome_exibicao}] Labels distintos: "
            f"{quantidade_labels}"
        )

        # No modo de teste, utiliza somente 10 observações
        # em cada etapa.
        n_gen_real = (
            10
            if MODO_TESTE
            else n_gen
        )

        n_ass_real = (
            10
            if MODO_TESTE
            else n_ass
        )

        quantidade_necessaria = (
            n_gen_real
            + n_ass_real
        )

        if len(df_total) < quantidade_necessaria:
            raise ValueError(
                f"{nome_exibicao}: a base unificada possui "
                f"{len(df_total)} documentos únicos, mas são "
                f"necessários pelo menos {quantidade_necessaria}: "
                f"{n_gen_real} para geração e "
                f"{n_ass_real} para atribuição."
            )

        # Embaralha toda a base uma única vez.
        # O random_state garante reprodutibilidade.
        df_total = (
            df_total
            .sample(
                frac=1,
                random_state=42,
            )
            .reset_index(drop=True)
        )

        # Seleciona os primeiros registros para geração.
        df_gen = (
            df_total
            .iloc[:n_gen_real]
            .copy()
            .reset_index(drop=True)
        )

        # Seleciona os registros seguintes para atribuição.
        # Como os intervalos não se sobrepõem, nenhum documento
        # de geração será utilizado na atribuição.
        inicio_atribuicao = n_gen_real

        fim_atribuicao = (
            n_gen_real
            + n_ass_real
        )

        df_ass = (
            df_total
            .iloc[
                inicio_atribuicao:
                fim_atribuicao
            ]
            .copy()
            .reset_index(drop=True)
        )

        # Validação adicional para garantir que não existe
        # sobreposição textual entre as duas amostras.
        textos_geracao = set(
            df_gen["text"]
        )

        textos_atribuicao = set(
            df_ass["text"]
        )

        documentos_sobrepostos = (
            textos_geracao
            .intersection(textos_atribuicao)
        )

        if documentos_sobrepostos:
            raise RuntimeError(
                f"{nome_exibicao}: foram encontrados "
                f"{len(documentos_sobrepostos)} documentos presentes "
                f"simultaneamente na geração e na atribuição."
            )

        # Valida novamente os labels nas duas amostras.
        labels_geracao = (
            df_gen["label_name"]
            .nunique(dropna=True)
        )

        labels_atribuicao = (
            df_ass["label_name"]
            .nunique(dropna=True)
        )

        if labels_geracao <= 1:
            raise ValueError(
                f"{nome_exibicao}: a amostra de geração possui "
                f"somente {labels_geracao} label distinta."
            )

        if labels_atribuicao <= 1:
            raise ValueError(
                f"{nome_exibicao}: a amostra de atribuição possui "
                f"somente {labels_atribuicao} label distinta."
            )

        pasta = Path(
            "datasets/experimento"
        )

        pasta.mkdir(
            parents=True,
            exist_ok=True,
        )

        path_gen = (
            pasta
            / f"{prefixo}_gen.jsonl"
        )

        path_ass = (
            pasta
            / f"{prefixo}_ass.jsonl"
        )

        # A coluna origem é usada apenas nas validações internas.
        # Os arquivos do experimento mantêm somente o formato
        # esperado pelo pipeline.
        df_gen[
            [
                "text",
                "label_name",
            ]
        ].to_json(
            path_gen,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        df_ass[
            [
                "text",
                "label_name",
            ]
        ].to_json(
            path_ass,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        print(
            f"[{nome_exibicao}] Preparação concluída!"
        )

        print(
            f"[{nome_exibicao}] Geração: "
            f"{len(df_gen)} documentos | "
            f"{labels_geracao} labels"
        )

        print(
            f"[{nome_exibicao}] Atribuição: "
            f"{len(df_ass)} documentos | "
            f"{labels_atribuicao} labels"
        )

        print(
            f"[{nome_exibicao}] Sobreposição entre as amostras: "
            f"{len(documentos_sobrepostos)}"
        )

        print(
            f"[{nome_exibicao}] Arquivo de geração: "
            f"{path_gen}"
        )

        print(
            f"[{nome_exibicao}] Arquivo de atribuição: "
            f"{path_ass}"
        )

        return (
            str(path_gen),
            str(path_ass),
            len(df_gen),
            len(df_ass),
        )

    @staticmethod
    def preparar_bills():
        return PreparadorDatasets._preparar_dataset(
            prefixo="bills", nome_exibicao="BILLS", n_gen=1000, n_ass=15242
        )

    @staticmethod
    def preparar_wiki():
        return PreparadorDatasets._preparar_dataset(
            prefixo="wiki", nome_exibicao="WIKI", n_gen=1100, n_ass=8024
        )

class PipelineMestrado:
    def __init__(self, model_name="gemma2:9b"):
        self.api = "openai"
        self.model_name = model_name

    @staticmethod
    def _ler_topicos_de_txt(caminho):
        padrao = re.compile(
            r"^\[(\d+)\]\s+(.+?)(?:\s+\(Count:\s*\d+\))?\s*:\s*(.*)$"
        )
        topicos = {}
        with open(caminho, "r", encoding="utf-8") as arquivo:
            for numero_linha, linha in enumerate(arquivo, start=1):
                conteudo = linha.strip()
                if not conteudo:
                    continue
                match = padrao.match(conteudo)
                if not match:
                    continue
                nivel, nome, descricao = match.groups()
                chave = f"[{nivel}] {nome.strip()}"
                topicos[chave] = {
                    "lvl": nivel,
                    "nome": nome.strip(),
                    "desc": descricao.strip() or "Descrição não informada",
                }
        return topicos

    def _salvar_topicos_para_atribuicao(self, arquivo_topicos, arquivo_saida):
        if not os.path.exists(arquivo_topicos) or os.path.getsize(arquivo_topicos) == 0:
            raise FileNotFoundError(
                f"Arquivo de tópicos ausente ou vazio: '{arquivo_topicos}'"
            )

        topicos = self._ler_topicos_de_txt(arquivo_topicos)
        if not topicos:
            raise ValueError(
                f"Nenhum tópico válido foi encontrado em '{arquivo_topicos}'."
            )

        linhas = [
            f"[{info['lvl']}] {info['nome']} (Count: 0): {info['desc']}"
            for info in topicos.values()
        ]
        with open(arquivo_saida, "w", encoding="utf-8") as arquivo:
            arquivo.write("\n".join(linhas))
        return arquivo_saida, len(topicos)

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
                verbose=False,
            )
        qtd_topicos = len(self._ler_topicos_de_txt(topic_file))
        tempo = time.time() - inicio
        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos únicos encontrados: {qtd_topicos}"
        )
        return qtd_topicos

    def refinar_topicos(
        self,
        prompt_file,
        generation_file,
        topic_file,
        out_file,
        updated_file,
        mapping_file,
    ):
        print(" -> Iniciando Fase 2: Refinamento de Tópicos...")
        inicio = time.time()
    
        # Conta os tópicos originais da fase de geração.
        qtd_antes = len(
            self._ler_topicos_de_txt(topic_file)
        )
    
        if qtd_antes == 0:
            raise ValueError(
                f"Nenhum tópico válido foi encontrado no arquivo "
                f"original: '{topic_file}'"
            )
    
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
                mapping_file=mapping_file,
            )
    
        # O TopicGPT salva a árvore final refinada em out_file.
        if not os.path.exists(out_file):
            raise FileNotFoundError(
                f"O refinamento não criou o arquivo esperado: "
                f"'{out_file}'"
            )
    
        if os.path.getsize(out_file) == 0:
            raise ValueError(
                f"O arquivo refinado foi criado vazio: "
                f"'{out_file}'"
            )
    
        # Mantém os nomes esperados pela rotina:
        # b_top_refinados_limpo.txt ou w_top_refinados_limpo.txt.
        arquivo_limpo = updated_file.replace(
            ".json",
            "_limpo.txt",
        )
    
        # A lista limpa deve ser criada a partir de out_file,
        # que contém a árvore refinada, e não de topic_file.
        arquivo_limpo, qtd_depois = (
            self._salvar_topicos_para_atribuicao(
                out_file,
                arquivo_limpo,
            )
        )
    
        if qtd_depois > qtd_antes:
            raise RuntimeError(
                "O refinamento aumentou a quantidade de tópicos "
                f"({qtd_antes} -> {qtd_depois}). "
                "Isso não é esperado. Verifique a saída em "
                f"'{out_file}'."
            )
    
        quantidade_fundida_ou_removida = (
            qtd_antes - qtd_depois
        )
    
        quantidade_mapeamentos = 0
    
        if os.path.exists(mapping_file):
            try:
                with open(
                    mapping_file,
                    "r",
                    encoding="utf-8",
                ) as arquivo:
                    mapeamentos = json.load(arquivo)
    
                quantidade_mapeamentos = sum(
                    1
                    for topico_original, topico_novo
                    in mapeamentos.items()
                    if topico_original != topico_novo
                )
    
            except (
                json.JSONDecodeError,
                OSError,
                AttributeError,
            ):
                quantidade_mapeamentos = 0
    
        tempo = time.time() - inicio
    
        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos: {qtd_antes} -> {qtd_depois} | "
            f"Redução: {quantidade_fundida_ou_removida} | "
            f"Mapeamentos alterados: {quantidade_mapeamentos}"
        )
    
        print(
            f"    [✔] Árvore refinada original: {out_file}"
        )
    
        print(
            f"    [✔] Lista refinada para atribuição: "
            f"{arquivo_limpo}"
        )
    
        return qtd_depois

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

        # Se já for o TXT refinado, usa diretamente. Caso contrário, converte.
        if topic_file.endswith(".txt"):
            topic_file_pronto = topic_file
            quantidade_topicos = len(self._ler_topicos_de_txt(topic_file))
        else:
            topic_file_pronto = topic_file.replace(".json", "_limpo.txt")
            topic_file_pronto, quantidade_topicos = self._salvar_topicos_para_atribuicao(
                topic_file, topic_file_pronto
            )

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
                    verbose=False,
                )
        except Exception as erro:
            raise RuntimeError(
                "A fase 3 falhou durante a atribuição de tópicos.\n"
                f"Dataset: {data_file}\n"
                f"Prompt: {prompt_file}\n"
                f"Tópicos: {topic_file_pronto}\n"
                f"Quantidade: {quantidade_topicos}\n"
                f"Erro original: {type(erro).__name__}: {erro}"
            ) from erro

        if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
            raise ValueError(f"A fase 3 não criou uma saída válida: '{out_file}'")
        tempo = time.time() - inicio
        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos disponíveis: {quantidade_topicos} | Saída: {out_file}"
        )

class AvaliadorMetricas:
    @staticmethod
    def calcular_purity(y_true, y_pred):
        matriz = contingency_matrix(
            y_true,
            y_pred,
        )

        return (
            np.sum(np.amax(matriz, axis=0))
            / np.sum(matriz)
        )

    @staticmethod
    def calcular_inverse_purity(y_true, y_pred):
        matriz = contingency_matrix(
            y_true,
            y_pred,
        )

        return (
            np.sum(np.amax(matriz, axis=1))
            / np.sum(matriz)
        )

    @staticmethod
    def _normalizar_topico(texto):
        texto = str(texto).strip().lower()

        texto = re.sub(
            r"\s+",
            " ",
            texto,
        )

        texto = texto.strip(
            " \t\r\n:;,.\"'`()[]"
        )

        return texto

    @staticmethod
    def _carregar_topicos_validos(topic_file):
        if not os.path.exists(topic_file):
            raise FileNotFoundError(
                f"Arquivo de tópicos não encontrado: "
                f"'{topic_file}'"
            )

        topicos_validos = {}

        padrao = re.compile(
            r"^\[\d+\]\s+"
            r"(.+?)"
            r"(?:\s+\(Count:\s*\d+\))?"
            r"\s*:"
        )

        with open(
            topic_file,
            "r",
            encoding="utf-8",
        ) as arquivo:
            for linha in arquivo:
                conteudo = linha.strip()

                if not conteudo:
                    continue

                match = padrao.match(conteudo)

                if not match:
                    continue

                nome_original = (
                    match.group(1).strip()
                )

                nome_normalizado = (
                    AvaliadorMetricas
                    ._normalizar_topico(
                        nome_original
                    )
                )

                topicos_validos[
                    nome_normalizado
                ] = nome_original

        if not topicos_validos:
            raise ValueError(
                f"Nenhum tópico válido foi encontrado em "
                f"'{topic_file}'."
            )

        return topicos_validos

    @staticmethod
    def _extrair_topico_valido(
        resposta,
        topicos_validos,
    ):
        if not isinstance(resposta, str):
            return None

        padrao_topico = re.compile(
            r"\[\d+\]\s*"
            r"([^:\n]+)"
        )

        candidatos = padrao_topico.findall(
            resposta
        )

        for candidato in candidatos:
            candidato = re.sub(
                r"\s+\(Count:\s*\d+\)\s*$",
                "",
                candidato,
                flags=re.IGNORECASE,
            )

            candidato_normalizado = (
                AvaliadorMetricas
                ._normalizar_topico(
                    candidato
                )
            )

            if (
                candidato_normalizado
                in topicos_validos
            ):
                return candidato_normalizado

        resposta_normalizada = (
            AvaliadorMetricas
            ._normalizar_topico(
                resposta
            )
        )

        topicos_ordenados = sorted(
            topicos_validos.keys(),
            key=len,
            reverse=True,
        )

        for topico in topicos_ordenados:
            padrao_nome = (
                r"(?<!\w)"
                + re.escape(topico)
                + r"(?!\w)"
            )

            if re.search(
                padrao_nome,
                resposta_normalizada,
            ):
                return topico

        return None

    @staticmethod
    def avaliar_resultados(
        caminho_arquivo_json,
        dataset_name,
        topic_file=None,
    ):
        if topic_file is None:
            if dataset_name.upper() == "WIKI":
                topic_file = (
                    "resultados/"
                    "w_top_refinados_limpo.txt"
                )

            elif dataset_name.upper() == "BILLS":
                topic_file = (
                    "resultados/"
                    "b_top_refinados_limpo.txt"
                )

            else:
                raise ValueError(
                    "Informe topic_file para datasets "
                    "diferentes de WIKI e BILLS."
                )

        topicos_validos = (
            AvaliadorMetricas
            ._carregar_topicos_validos(
                topic_file
            )
        )

        y_true = []
        y_pred = []

        labels_originais_set = set()
        topicos_preditos_set = set()

        respostas_sem_topico_valido = 0
        respostas_sem_conteudo = 0

        with open(
            caminho_arquivo_json,
            "r",
            encoding="utf-8",
        ) as arquivo:
            try:
                dados = json.load(arquivo)

            except json.JSONDecodeError:
                arquivo.seek(0)

                dados = [
                    json.loads(linha)
                    for linha in arquivo
                    if linha.strip()
                ]

        for item in dados:
            verdadeiro = item.get(
                "label_name"
            )

            resposta = item.get(
                "responses",
                "",
            )

            if verdadeiro is None:
                continue

            if not isinstance(resposta, str):
                respostas_sem_conteudo += 1
                continue

            if not resposta.strip():
                respostas_sem_conteudo += 1
                continue

            topico_predito = (
                AvaliadorMetricas
                ._extrair_topico_valido(
                    resposta=resposta,
                    topicos_validos=(
                        topicos_validos
                    ),
                )
            )

            if topico_predito is None:
                respostas_sem_topico_valido += 1
                continue

            verdadeiro_limpo = (
                AvaliadorMetricas
                ._normalizar_topico(
                    verdadeiro
                )
            )

            y_true.append(
                verdadeiro_limpo
            )

            y_pred.append(
                topico_predito
            )

            labels_originais_set.add(
                verdadeiro_limpo
            )

            topicos_preditos_set.add(
                topico_predito
            )

        if not y_true:
            raise ValueError(
                "Nenhuma observação válida foi encontrada "
                "para o cálculo das métricas."
            )

        if (
            "n/a" in labels_originais_set
            or len(labels_originais_set) <= 1
        ):
            print(
                " -> Aviso: ground truth válido "
                "não foi detectado."
            )

            return {
                "NMI": None,
                "ARI": None,
                "HMP": None,
                "Purity": None,
                "Inverse_Purity": None,
                "Amostras_Validadas": len(y_true),
                "Topicos_Disponiveis": len(
                    topicos_validos
                ),
                "Topicos_Utilizados": len(
                    topicos_preditos_set
                ),
                "Topicos_Ground_Truth": 0,
                "Respostas_Sem_Topico_Valido": (
                    respostas_sem_topico_valido
                ),
                "Respostas_Sem_Conteudo": (
                    respostas_sem_conteudo
                ),
            }

        nmi = normalized_mutual_info_score(
            y_true,
            y_pred,
        )

        ari = adjusted_rand_score(
            y_true,
            y_pred,
        )

        purity = (
            AvaliadorMetricas
            .calcular_purity(
                y_true,
                y_pred,
            )
        )

        inverse_purity = (
            AvaliadorMetricas
            .calcular_inverse_purity(
                y_true,
                y_pred,
            )
        )

        if (
            purity + inverse_purity
        ) == 0:
            hmp = 0

        else:
            hmp = (
                2
                * purity
                * inverse_purity
                / (
                    purity
                    + inverse_purity
                )
            )

        print(
            f"\n--- Resultados da Avaliação: "
            f"{dataset_name} ---"
        )

        print(
            f"NMI: {nmi:.4f} | "
            f"ARI: {ari:.4f} | "
            f"HMP: {hmp:.4f}"
        )

        print(
            f"Tópicos disponíveis: "
            f"{len(topicos_validos)} | "
            f"Tópicos efetivamente utilizados: "
            f"{len(topicos_preditos_set)}"
        )

        print(
            f"Amostras validadas: "
            f"{len(y_true)} | "
            f"Respostas sem tópico válido: "
            f"{respostas_sem_topico_valido} | "
            f"Respostas vazias: "
            f"{respostas_sem_conteudo}"
        )

        return {
            "NMI": nmi,
            "ARI": ari,
            "HMP": hmp,
            "Purity": purity,
            "Inverse_Purity": inverse_purity,
            "Amostras_Validadas": len(y_true),
            "Topicos_Disponiveis": len(
                topicos_validos
            ),
            "Topicos_Utilizados": len(
                topicos_preditos_set
            ),
            "Topicos_Ground_Truth": len(
                labels_originais_set
            ),
            "Respostas_Sem_Topico_Valido": (
                respostas_sem_topico_valido
            ),
            "Respostas_Sem_Conteudo": (
                respostas_sem_conteudo
            ),
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
    pipeline.atribuir_topicos(file_bills_ass, "prompts/assignment.txt", "resultados/b_atribuida.json", "resultados/b_top_refinados_limpo.txt")
    
    metricas_bills = AvaliadorMetricas.avaliar_resultados("resultados/b_atribuida.json", "BILLS")
    registrar_log("BILLS", metricas_bills, n_gen_b, n_ass_b)

    # 2. PROCESSAR WIKITEXT
    print("\n" + "="*50 + "\nDATASET 2: WIKI\n" + "="*50)
    file_wiki_gen, file_wiki_ass, n_gen_w, n_ass_w = PreparadorDatasets.preparar_wiki()
    
    qtd_gen_wiki = pipeline.gerar_topicos(file_wiki_gen, "prompts/generation_1.txt", "prompts/seed_1.md", "resultados/w_geracao.json", "resultados/w_topicos.json")
    qtd_ref_wiki = pipeline.refinar_topicos("prompts/refinement.txt", "resultados/w_geracao.json", "resultados/w_topicos.json", "resultados/w_refinamento.json", "resultados/w_top_refinados.json", "resultados/w_map.json")
    pipeline.atribuir_topicos(file_wiki_ass, "prompts/assignment.txt", "resultados/w_atribuida.json", "resultados/w_top_refinados_limpo.txt")
    
    metricas_wiki = AvaliadorMetricas.avaliar_resultados("resultados/w_atribuida.json", "WIKI")
    registrar_log("WIKI", metricas_wiki, n_gen_w, n_ass_w)
    
    print("\n🎉 === EXPERIMENTO FINALIZADO COM SUCESSO! === 🎉")