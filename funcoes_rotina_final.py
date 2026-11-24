import warnings

# Silencia warnings de ambiente/dependências antes dos imports das bibliotecas
warnings.filterwarnings("ignore", message=r".*IProgress not found.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.api_core.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.generativeai.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"topicgpt_python.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*google\.generativeai.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*non-supported Python version.*")

import os
import json
import re
import sys
import time
import functools
import contextlib
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
import openai
from dotenv import load_dotenv

# ==============================================================================
# CONTROLE DE SAÍDA
# ==============================================================================
@contextlib.contextmanager
def silenciar_saida_detalhada(ativo=True):
    """
    Silencia temporariamente prints, barras tqdm e logs verbosos.
    Quando ativo=False (ex.: MODO_TESTE=True), mantém a saída visível para diagnóstico.
    """
    if not ativo:
        yield
        return

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

    root_logger = logging.getLogger()
    nivel_root_anterior = root_logger.level
    root_logger.setLevel(logging.CRITICAL + 1)

    try:
        with open(os.devnull, "w", encoding="utf-8") as destino_nulo:
            with contextlib.redirect_stdout(destino_nulo), \
                 contextlib.redirect_stderr(destino_nulo):
                yield

    finally:
        root_logger.setLevel(nivel_root_anterior)

        for nome_logger, estado in estados_anteriores.items():
            logger = logging.getLogger(nome_logger)
            logger.setLevel(estado["level"])
            logger.disabled = estado["disabled"]
            logger.propagate = estado["propagate"]

# ==============================================================================
# CONFIGURAÇÕES GERAIS DO EXPERIMENTO
# ==============================================================================

# Mude para False quando validar o teste de 10 documentos e for rodar o experimento completo
MODO_TESTE = False

# Configurações da API / Proxy UnB
ARQUIVO_ENV = "env.env"
NOME_VAR_CHAVE = "UNB_AK"
BASE_URL_PROXY = "https://ia-router.unb.br/"
MODELO_LLM = "UnB-Qwen/qwen3-8-27b"

# Parâmetros oficiais do experimento para salvar no Log
HIPERPARAMETROS = {
    "max_tokens": 300,
    "temperature": 0.0,
    "top_p": 1.0,  # Ajustado de 0.0 para 1.0 (equivalente com temperature=0.0; exigido pelo vLLM/LiteLLM)
    "auto_correction_limit": 10,
    "min_doc_frequency_bills": 10,
    "min_doc_frequency_wiki": 5,
    "max_words_per_doc": 250,
}


class ConfiguradorUnB:
    _estrategia_thinking = None
    _min_max_tokens_thinking = 2048

    @staticmethod
    def _limpar_conteudo(texto):
        if not isinstance(texto, str):
            return ""
        # Remove eventuais blocos <think>...</think> que vazem no content
        limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
        return limpo.strip()

    @classmethod
    def _ajustar_kwargs(cls, kwargs):
        # 1. Corrige top_p=0.0 -> 1.0 exigido pelo LiteLLM/vLLM
        if "top_p" in kwargs and kwargs["top_p"] is not None and kwargs["top_p"] <= 0:
            kwargs["top_p"] = 1.0

        # 2. Aplica a estratégia ativa para lidar com o modo thinking do Qwen3
        if cls._estrategia_thinking == "extra_body":
            extra_body = dict(kwargs.get("extra_body") or {})
            chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_kwargs
            kwargs["extra_body"] = extra_body

        elif cls._estrategia_thinking == "no_think":
            msgs = kwargs.get("messages")
            if isinstance(msgs, list) and msgs:
                novas_msgs = [dict(m) for m in msgs]
                for i in range(len(novas_msgs) - 1, -1, -1):
                    if (
                        novas_msgs[i].get("role") == "user"
                        and isinstance(novas_msgs[i].get("content"), str)
                    ):
                        if "/no_think" not in novas_msgs[i]["content"]:
                            novas_msgs[i]["content"] = (
                                novas_msgs[i]["content"].rstrip() + "\n/no_think"
                            )
                        break
                kwargs["messages"] = novas_msgs

        elif cls._estrategia_thinking == "high_max_tokens":
            atual = kwargs.get("max_tokens")
            if atual is None or atual < cls._min_max_tokens_thinking:
                kwargs["max_tokens"] = cls._min_max_tokens_thinking

        return kwargs

    @classmethod
    def _limpar_resposta(cls, response):
        try:
            if getattr(response, "choices", None):
                for choice in response.choices:
                    msg = getattr(choice, "message", None)
                    if msg is not None and isinstance(getattr(msg, "content", None), str):
                        msg.content = cls._limpar_conteudo(msg.content)
        except Exception:
            pass
        return response

    @classmethod
    def aplicar_patch_openai(cls):
        """
        Intercepta chamadas do SDK OpenAI feitas pelo topicgpt_python para:
        - converter top_p <= 0 em 1.0;
        - aplicar a estratégia detectada para modelos com reasoning/thinking (Qwen3);
        - limpar tags <think>...</think> residuais na resposta.
        """
        from openai.resources.chat.completions import Completions, AsyncCompletions

        if not getattr(Completions.create, "_unb_patched", False):
            original_create = Completions.create

            @functools.wraps(original_create)
            def patched_create(self, *args, **kwargs):
                cls._ajustar_kwargs(kwargs)
                response = original_create(self, *args, **kwargs)
                return cls._limpar_resposta(response)

            patched_create._unb_patched = True
            Completions.create = patched_create

        if not getattr(AsyncCompletions.create, "_unb_patched", False):
            original_async_create = AsyncCompletions.create

            @functools.wraps(original_async_create)
            async def patched_async_create(self, *args, **kwargs):
                cls._ajustar_kwargs(kwargs)
                response = await original_async_create(self, *args, **kwargs)
                return cls._limpar_resposta(response)

            patched_async_create._unb_patched = True
            AsyncCompletions.create = patched_async_create

    @classmethod
    def setup_ambiente_topicgpt(
        cls,
        arquivo_env=ARQUIVO_ENV,
        nome_var_chave=NOME_VAR_CHAVE,
        base_url=BASE_URL_PROXY,
    ):
        load_dotenv(arquivo_env, override=True)
        chave = os.getenv(nome_var_chave)

        if not chave:
            raise ValueError(
                f"ERRO: a variável '{nome_var_chave}' não foi carregada a partir de '{arquivo_env}'."
            )

        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = chave

        cls.aplicar_patch_openai()
        print(f"Ambiente configurado! topicgpt_python apontará para: {base_url} (patch UnB ativo)")
        return chave

    @classmethod
    def testar_conexao(
        cls,
        model_name=MODELO_LLM,
        base_url=BASE_URL_PROXY,
    ):
        chave = os.environ.get("OPENAI_API_KEY")
        if not chave:
            raise ValueError(
                "OPENAI_API_KEY não configurada. Execute setup_ambiente_topicgpt() primeiro."
            )

        client = openai.OpenAI(api_key=chave, base_url=base_url)

        estrategias = [
            ("extra_body", "desabilitar thinking via chat_template_kwargs"),
            ("no_think", "desabilitar thinking via sufixo /no_think"),
            (
                "high_max_tokens",
                f"manter thinking ativo com max_tokens={cls._min_max_tokens_thinking}",
            ),
        ]

        print(f"Testando conexão e modo de resposta no proxy da UnB ({model_name})...")

        for estrategia, descricao in estrategias:
            cls._estrategia_thinking = estrategia
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Responda apenas 'OK'."}],
                    temperature=0.0,
                    top_p=0.0,
                    max_tokens=64,
                )
                choice = response.choices[0]
                msg = choice.message
                conteudo = msg.content
                reasoning = getattr(msg, "reasoning_content", None) or (
                    getattr(msg, "model_extra", None) or {}
                ).get("reasoning_content")

                if isinstance(conteudo, str) and conteudo.strip():
                    HIPERPARAMETROS["estrategia_thinking"] = estrategia
                    if estrategia == "high_max_tokens":
                        HIPERPARAMETROS["max_tokens"] = cls._min_max_tokens_thinking
                    print(
                        f"Conexão validada com sucesso! Estratégia ativa: '{estrategia}' "
                        f"({descricao}) | Resposta: {conteudo.strip()!r}"
                    )
                    return conteudo.strip()

                print(
                    f" -> Estratégia '{estrategia}' retornou content vazio/None "
                    f"(finish_reason={choice.finish_reason}, "
                    f"reasoning_content_presente={bool(reasoning)}). Tentando próxima..."
                )
            except Exception as erro:
                print(
                    f" -> Estratégia '{estrategia}' falhou ({type(erro).__name__}: {erro}). "
                    f"Tentando próxima..."
                )

        cls._estrategia_thinking = None
        raise RuntimeError(
            f"O modelo '{model_name}' não retornou texto em 'message.content' em nenhuma "
            f"estratégia testada. Verifique os parâmetros aceitos pelo proxy da UnB."
        )

# Alias mantido apenas por compatibilidade caso algum script externo importe o nome antigo
ConfiguradorOllama = ConfiguradorUnB


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

        caminho_train = PreparadorDatasets._baixar_arquivo_drive(f"{prefixo}_train")
        caminho_test = PreparadorDatasets._baixar_arquivo_drive(f"{prefixo}_test")

        df_train = PreparadorDatasets._ler_jsonl_com_labels(caminho_train, nome_exibicao)
        df_test = PreparadorDatasets._ler_jsonl_com_labels(caminho_test, nome_exibicao)

        print(
            f"[{nome_exibicao}] Registros antes da união | "
            f"Train: {len(df_train)} | "
            f"Test: {len(df_test)}"
        )

        df_train = df_train.copy()
        df_test = df_test.copy()

        df_train["origem"] = "train"
        df_test["origem"] = "test"

        df_total = pd.concat([df_train, df_test], ignore_index=True)
        quantidade_antes_deduplicacao = len(df_total)

        df_total = (
            df_total
            .drop_duplicates(subset=["text"], keep="first")
            .reset_index(drop=True)
        )

        quantidade_duplicados = quantidade_antes_deduplicacao - len(df_total)

        print(f"[{nome_exibicao}] Base total após a união: {len(df_total)} documentos únicos")
        print(f"[{nome_exibicao}] Duplicidades removidas: {quantidade_duplicados}")

        quantidade_labels = df_total["label_name"].nunique(dropna=True)

        if quantidade_labels <= 1:
            raise ValueError(
                f"{nome_exibicao}: a base total possui somente "
                f"{quantidade_labels} label distinta."
            )

        print(f"[{nome_exibicao}] Labels distintos: {quantidade_labels}")

        n_gen_real = 10 if MODO_TESTE else n_gen
        n_ass_real = 10 if MODO_TESTE else n_ass
        quantidade_necessaria = n_gen_real + n_ass_real

        if len(df_total) < quantidade_necessaria:
            raise ValueError(
                f"{nome_exibicao}: a base unificada possui "
                f"{len(df_total)} documentos únicos, mas são "
                f"necessários pelo menos {quantidade_necessaria}: "
                f"{n_gen_real} para geração e "
                f"{n_ass_real} para atribuição."
            )

        df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

        df_gen = df_total.iloc[:n_gen_real].copy().reset_index(drop=True)

        inicio_atribuicao = n_gen_real
        fim_atribuicao = n_gen_real + n_ass_real
        df_ass = df_total.iloc[inicio_atribuicao:fim_atribuicao].copy().reset_index(drop=True)

        textos_geracao = set(df_gen["text"])
        textos_atribuicao = set(df_ass["text"])
        documentos_sobrepostos = textos_geracao.intersection(textos_atribuicao)

        if documentos_sobrepostos:
            raise RuntimeError(
                f"{nome_exibicao}: foram encontrados "
                f"{len(documentos_sobrepostos)} documentos presentes "
                f"simultaneamente na geração e na atribuição."
            )

        labels_geracao = df_gen["label_name"].nunique(dropna=True)
        labels_atribuicao = df_ass["label_name"].nunique(dropna=True)

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

        pasta = Path("datasets/experimento")
        pasta.mkdir(parents=True, exist_ok=True)

        path_gen = pasta / f"{prefixo}_gen.jsonl"
        path_ass = pasta / f"{prefixo}_ass.jsonl"

        df_gen[["text", "label_name"]].to_json(
            path_gen,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        df_ass[["text", "label_name"]].to_json(
            path_ass,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        print(f"[{nome_exibicao}] Preparação concluída!")
        print(f"[{nome_exibicao}] Geração: {len(df_gen)} documentos | {labels_geracao} labels")
        print(f"[{nome_exibicao}] Atribuição: {len(df_ass)} documentos | {labels_atribuicao} labels")
        print(f"[{nome_exibicao}] Sobreposição entre as amostras: {len(documentos_sobrepostos)}")
        print(f"[{nome_exibicao}] Arquivo de geração: {path_gen}")
        print(f"[{nome_exibicao}] Arquivo de atribuição: {path_ass}")

        return str(path_gen), str(path_ass), len(df_gen), len(df_ass)

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
    def __init__(self, model_name=MODELO_LLM):
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
        with silenciar_saida_detalhada(ativo=not MODO_TESTE):
            generate_topic_lvl1(
                api=self.api,
                model=self.model_name,
                data=data_file,
                prompt_file=prompt_file,
                seed_file=seed_file,
                out_file=out_file,
                topic_file=topic_file,
                verbose=MODO_TESTE,
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

        qtd_antes = len(self._ler_topicos_de_txt(topic_file))

        if qtd_antes == 0:
            raise ValueError(
                f"Nenhum tópico válido foi encontrado no arquivo "
                f"original: '{topic_file}'"
            )

        with silenciar_saida_detalhada(ativo=not MODO_TESTE):
            from topicgpt_python import refine_topics

            refine_topics(
                api=self.api,
                model=self.model_name,
                prompt_file=prompt_file,
                generation_file=generation_file,
                topic_file=topic_file,
                out_file=out_file,
                updated_file=updated_file,
                verbose=MODO_TESTE,
                remove=True,
                mapping_file=mapping_file,
            )

        if not os.path.exists(out_file):
            raise FileNotFoundError(
                f"O refinamento não criou o arquivo esperado: '{out_file}'"
            )

        if os.path.getsize(out_file) == 0:
            raise ValueError(
                f"O arquivo refinado foi criado vazio: '{out_file}'"
            )

        arquivo_limpo = updated_file.replace(".json", "_limpo.txt")

        arquivo_limpo, qtd_depois = self._salvar_topicos_para_atribuicao(
            out_file,
            arquivo_limpo,
        )

        if qtd_depois > qtd_antes:
            raise RuntimeError(
                "O refinamento aumentou a quantidade de tópicos "
                f"({qtd_antes} -> {qtd_depois}). "
                "Isso não é esperado. Verifique a saída em "
                f"'{out_file}'."
            )

        quantidade_fundida_ou_removida = qtd_antes - qtd_depois
        quantidade_mapeamentos = 0

        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, "r", encoding="utf-8") as arquivo:
                    mapeamentos = json.load(arquivo)

                quantidade_mapeamentos = sum(
                    1
                    for topico_original, topico_novo in mapeamentos.items()
                    if topico_original != topico_novo
                )
            except (json.JSONDecodeError, OSError, AttributeError):
                quantidade_mapeamentos = 0

        tempo = time.time() - inicio

        print(
            f"    [✔] Concluído em {tempo:.2f}s | "
            f"Tópicos: {qtd_antes} -> {qtd_depois} | "
            f"Redução: {quantidade_fundida_ou_removida} | "
            f"Mapeamentos alterados: {quantidade_mapeamentos}"
        )
        print(f"    [✔] Árvore refinada original: {out_file}")
        print(f"    [✔] Lista refinada para atribuição: {arquivo_limpo}")

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
            with silenciar_saida_detalhada(ativo=not MODO_TESTE):
                from topicgpt_python import assign_topics
                assign_topics(
                    api=self.api,
                    model=self.model_name,
                    data=data_file,
                    prompt_file=prompt_file,
                    out_file=out_file,
                    topic_file=topic_file_pronto,
                    verbose=MODO_TESTE,
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
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=0)) / np.sum(matriz)

    @staticmethod
    def calcular_inverse_purity(y_true, y_pred):
        matriz = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matriz, axis=1)) / np.sum(matriz)

    @staticmethod
    def _normalizar_topico(texto):
        texto = str(texto).strip().lower()
        texto = re.sub(r"\s+", " ", texto)
        texto = texto.strip(" \t\r\n:;,.\"'`()[]")
        return texto

    @staticmethod
    def _carregar_topicos_validos(topic_file):
        if not os.path.exists(topic_file):
            raise FileNotFoundError(f"Arquivo de tópicos não encontrado: '{topic_file}'")

        topicos_validos = {}
        padrao = re.compile(r"^\[\d+\]\s+(.+?)(?:\s+\(Count:\s*\d+\))?\s*:")

        with open(topic_file, "r", encoding="utf-8") as arquivo:
            for linha in arquivo:
                conteudo = linha.strip()
                if not conteudo:
                    continue
                match = padrao.match(conteudo)
                if not match:
                    continue
                nome_original = match.group(1).strip()
                nome_normalizado = AvaliadorMetricas._normalizar_topico(nome_original)
                topicos_validos[nome_normalizado] = nome_original

        if not topicos_validos:
            raise ValueError(f"Nenhum tópico válido foi encontrado em '{topic_file}'.")

        return topicos_validos

    @staticmethod
    def _extrair_topico_valido(resposta, topicos_validos):
        if not isinstance(resposta, str):
            return None

        padrao_topico = re.compile(r"\[\d+\]\s*([^:\n]+)")
        candidatos = padrao_topico.findall(resposta)

        for candidato in candidatos:
            candidato = re.sub(
                r"\s+\(Count:\s*\d+\)\s*$",
                "",
                candidato,
                flags=re.IGNORECASE,
            )
            candidato_normalizado = AvaliadorMetricas._normalizar_topico(candidato)
            if candidato_normalizado in topicos_validos:
                return candidato_normalizado

        resposta_normalizada = AvaliadorMetricas._normalizar_topico(resposta)
        topicos_ordenados = sorted(topicos_validos.keys(), key=len, reverse=True)

        for topico in topicos_ordenados:
            padrao_nome = r"(?<!\w)" + re.escape(topico) + r"(?!\w)"
            if re.search(padrao_nome, resposta_normalizada):
                return topico

        return None

    @staticmethod
    def avaliar_resultados(caminho_arquivo_json, dataset_name, topic_file=None):
        if topic_file is None:
            if dataset_name.upper() == "WIKI":
                topic_file = "resultados/w_top_refinados_limpo.txt"
            elif dataset_name.upper() == "BILLS":
                topic_file = "resultados/b_top_refinados_limpo.txt"
            else:
                raise ValueError("Informe topic_file para datasets diferentes de WIKI e BILLS.")

        topicos_validos = AvaliadorMetricas._carregar_topicos_validos(topic_file)

        y_true = []
        y_pred = []

        labels_originais_set = set()
        topicos_preditos_set = set()

        respostas_sem_topico_valido = 0
        respostas_sem_conteudo = 0

        with open(caminho_arquivo_json, "r", encoding="utf-8") as arquivo:
            try:
                dados = json.load(arquivo)
            except json.JSONDecodeError:
                arquivo.seek(0)
                dados = [json.loads(linha) for linha in arquivo if linha.strip()]

        for item in dados:
            verdadeiro = item.get("label_name")
            resposta = item.get("responses", "")

            if verdadeiro is None:
                continue

            if not isinstance(resposta, str) or not resposta.strip():
                respostas_sem_conteudo += 1
                continue

            topico_predito = AvaliadorMetricas._extrair_topico_valido(
                resposta=resposta,
                topicos_validos=topicos_validos,
            )

            if topico_predito is None:
                respostas_sem_topico_valido += 1
                continue

            verdadeiro_limpo = AvaliadorMetricas._normalizar_topico(verdadeiro)

            y_true.append(verdadeiro_limpo)
            y_pred.append(topico_predito)
            labels_originais_set.add(verdadeiro_limpo)
            topicos_preditos_set.add(topico_predito)

        if not y_true:
            raise ValueError(
                "Nenhuma observação válida foi encontrada para o cálculo das métricas."
            )

        if "n/a" in labels_originais_set or len(labels_originais_set) <= 1:
            print(" -> Aviso: ground truth válido não foi detectado.")
            return {
                "NMI": None,
                "ARI": None,
                "HMP": None,
                "Purity": None,
                "Inverse_Purity": None,
                "Amostras_Validadas": len(y_true),
                "Topicos_Disponiveis": len(topicos_validos),
                "Topicos_Utilizados": len(topicos_preditos_set),
                "Topicos_Ground_Truth": 0,
                "Respostas_Sem_Topico_Valido": respostas_sem_topico_valido,
                "Respostas_Sem_Conteudo": respostas_sem_conteudo,
            }

        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        purity = AvaliadorMetricas.calcular_purity(y_true, y_pred)
        inverse_purity = AvaliadorMetricas.calcular_inverse_purity(y_true, y_pred)

        hmp = 0 if (purity + inverse_purity) == 0 else (
            2 * purity * inverse_purity / (purity + inverse_purity)
        )

        print(f"\n--- Resultados da Avaliação: {dataset_name} ---")
        print(f"NMI: {nmi:.4f} | ARI: {ari:.4f} | HMP: {hmp:.4f}")
        print(
            f"Tópicos disponíveis: {len(topicos_validos)} | "
            f"Tópicos efetivamente utilizados: {len(topicos_preditos_set)}"
        )
        print(
            f"Amostras validadas: {len(y_true)} | "
            f"Respostas sem tópico válido: {respostas_sem_topico_valido} | "
            f"Respostas vazias: {respostas_sem_conteudo}"
        )

        return {
            "NMI": nmi,
            "ARI": ari,
            "HMP": hmp,
            "Purity": purity,
            "Inverse_Purity": inverse_purity,
            "Amostras_Validadas": len(y_true),
            "Topicos_Disponiveis": len(topicos_validos),
            "Topicos_Utilizados": len(topicos_preditos_set),
            "Topicos_Ground_Truth": len(labels_originais_set),
            "Respostas_Sem_Topico_Valido": respostas_sem_topico_valido,
            "Respostas_Sem_Conteudo": respostas_sem_conteudo,
        }


def registrar_log(dataset, metricas, n_gen, n_ass, modelo=MODELO_LLM):
    log_file = "experimento_metricas_geral.json"
    registro = {
        "dataset": dataset,
        "modelo": modelo,
        "base_url": BASE_URL_PROXY,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modo_teste": MODO_TESTE,
        "amostra_geracao": n_gen,
        "amostra_atribuicao": n_ass,
        "parametros_llm": HIPERPARAMETROS,
        "metricas": metricas,
    }

    historico = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            historico = json.load(f)

    historico.append(registro)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=4, ensure_ascii=False)
    print(f"\n[💾] Log salvo com sucesso em '{log_file}' (Fora da pasta resultados).")


# ==============================================================================
# FLUXO DE EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print(f"=== INICIANDO EXPERIMENTO TOPICGPT {'(MODO TESTE)' if MODO_TESTE else '(MODO COMPLETO)'} ===")
    print(f"Modelo selecionado: {MODELO_LLM}")

    os.makedirs("resultados", exist_ok=True)
    ConfiguradorUnB.setup_ambiente_topicgpt()
    ConfiguradorUnB.testar_conexao(model_name=MODELO_LLM)

    pipeline = PipelineMestrado(model_name=MODELO_LLM)

    # 1. PROCESSAR BILLS
    print("\n" + "=" * 50 + "\nDATASET 1: CONGRESSIONAL BILLS\n" + "=" * 50)
    file_bills_gen, file_bills_ass, n_gen_b, n_ass_b = PreparadorDatasets.preparar_bills()

    qtd_gen_bills = pipeline.gerar_topicos(
        file_bills_gen,
        "prompts/generation_1.txt",
        "prompts/seed_1.md",
        "resultados/b_geracao.json",
        "resultados/b_topicos.json",
    )
    qtd_ref_bills = pipeline.refinar_topicos(
        "prompts/refinement.txt",
        "resultados/b_geracao.json",
        "resultados/b_topicos.json",
        "resultados/b_refinamento.json",
        "resultados/b_top_refinados.json",
        "resultados/b_map.json",
    )
    pipeline.atribuir_topicos(
        file_bills_ass,
        "prompts/assignment.txt",
        "resultados/b_atribuida.json",
        "resultados/b_top_refinados_limpo.txt",
    )

    metricas_bills = AvaliadorMetricas.avaliar_resultados("resultados/b_atribuida.json", "BILLS")
    registrar_log("BILLS", metricas_bills, n_gen_b, n_ass_b, modelo=MODELO_LLM)

    # 2. PROCESSAR WIKITEXT
    print("\n" + "=" * 50 + "\nDATASET 2: WIKI\n" + "=" * 50)
    file_wiki_gen, file_wiki_ass, n_gen_w, n_ass_w = PreparadorDatasets.preparar_wiki()

    qtd_gen_wiki = pipeline.gerar_topicos(
        file_wiki_gen,
        "prompts/generation_1.txt",
        "prompts/seed_1.md",
        "resultados/w_geracao.json",
        "resultados/w_topicos.json",
    )
    qtd_ref_wiki = pipeline.refinar_topicos(
        "prompts/refinement.txt",
        "resultados/w_geracao.json",
        "resultados/w_topicos.json",
        "resultados/w_refinamento.json",
        "resultados/w_top_refinados.json",
        "resultados/w_map.json",
    )
    pipeline.atribuir_topicos(
        file_wiki_ass,
        "prompts/assignment.txt",
        "resultados/w_atribuida.json",
        "resultados/w_top_refinados_limpo.txt",
    )

    metricas_wiki = AvaliadorMetricas.avaliar_resultados("resultados/w_atribuida.json", "WIKI")
    registrar_log("WIKI", metricas_wiki, n_gen_w, n_ass_w, modelo=MODELO_LLM)

    print("\n🎉 === EXPERIMENTO FINALIZADO COM SUCESSO! === 🎉")