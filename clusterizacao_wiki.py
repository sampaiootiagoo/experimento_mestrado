"""Clusterizacao do Wiki com BGE-M3, HDBSCAN e selecao de K no desenvolvimento."""

import gc
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from funcoes_rotina_final import PreparadorDatasets

try:
    import torch
    from FlagEmbedding import BGEM3FlagModel
except ImportError as exc:
    raise ImportError(
        "Instale: pip install -U FlagEmbedding torch transformers accelerate"
    ) from exc

try:
    import hdbscan
    from hdbscan.prediction import approximate_predict
except ImportError as exc:
    raise ImportError("Instale: pip install -U hdbscan") from exc


CONFIG = {
    "seed": 42,
    "pasta_saida": "resultados_wiki_bgem3",
    "modelo_embedding": "BAAI/bge-m3",
    "proporcao_teste": 0.30,
    "batch_size_gpu": 16,
    "batch_size_cpu": 4,
    "max_length": 1024,
    "reutilizar_embeddings": True,
    "n_componentes_pca": 50,
    "k_min": 5,
    "k_max": 60,
    "k_passo": 1,
    "amostra_silhouette_k": 5000,
    "amostra_silhouette_avaliacao": 5000,
    "algoritmos_particionais": ["kmeans", "minibatch_kmeans", "birch"],
    "hdbscan_min_cluster_size": [25, 50, 100, 200, 400],
    "hdbscan_min_samples": [5, 10, 20, 40],
    "hdbscan_selection_method": ["eom", "leaf"],
    "peso_cobertura_hdbscan": 1.0,
    "top_labels_por_cluster": 5,
    "top_documentos_por_cluster": 3,
}


def criar_pastas():
    raiz = Path(CONFIG["pasta_saida"])
    pastas = {nome: raiz / nome for nome in ["cache", "graficos", "listas", "tabelas"]}
    pastas["raiz"] = raiz
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def carregar_wiki():
    caminho_train = PreparadorDatasets._baixar_arquivo_drive("wiki_train")
    caminho_test = PreparadorDatasets._baixar_arquivo_drive("wiki_test")
    df_train = PreparadorDatasets._ler_jsonl_com_labels(caminho_train, "WIKI_TRAIN").copy()
    df_test = PreparadorDatasets._ler_jsonl_com_labels(caminho_test, "WIKI_TEST").copy()
    df_train["origem_original"] = "train"
    df_test["origem_original"] = "test"

    df = pd.concat([df_train, df_test], ignore_index=True)
    antes = len(df)
    df = df.dropna(subset=["text", "label_name"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label_name"] = df["label_name"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label_name"] != "")]
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df["document_id"] = np.arange(len(df))

    if df.empty or df["label_name"].nunique() < 2:
        raise ValueError("A base Wiki nao possui documentos/labels suficientes.")

    print(f"Wiki completo: {len(df):,} documentos")
    print(f"Duplicados/invalidos removidos: {antes - len(df):,}")
    print(f"Ground truths: {df['label_name'].nunique():,}")
    return df


def dividir_desenvolvimento_teste(df):
    contagens = df["label_name"].value_counts()
    labels_raros = set(contagens[contagens < 2].index)
    raros = df[df["label_name"].isin(labels_raros)]
    elegiveis = df[~df["label_name"].isin(labels_raros)]

    dev, teste = train_test_split(
        elegiveis,
        test_size=CONFIG["proporcao_teste"],
        random_state=CONFIG["seed"],
        stratify=elegiveis["label_name"],
    )
    if len(raros):
        dev = pd.concat([dev, raros], ignore_index=True)

    dev = dev.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    teste = teste.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    dev["particao_experimento"] = "desenvolvimento"
    teste["particao_experimento"] = "teste"
    print(f"Desenvolvimento: {len(dev):,}; teste: {len(teste):,}")
    return dev, teste


def carregar_modelo_bgem3():
    usa_gpu = torch.cuda.is_available()
    print(f"Dispositivo BGE-M3: {'CUDA' if usa_gpu else 'CPU'}")
    modelo = BGEM3FlagModel(
        CONFIG["modelo_embedding"], use_fp16=usa_gpu, devices=None
    )
    return modelo, usa_gpu


def gerar_embeddings(modelo, textos, caminho_cache, usa_gpu):
    if CONFIG["reutilizar_embeddings"] and caminho_cache.exists():
        emb = np.load(caminho_cache)
        if len(emb) == len(textos):
            print(f"Embeddings carregados do cache: {caminho_cache}")
            return emb
        print("Cache ignorado: a quantidade de documentos mudou.")

    batch = CONFIG["batch_size_gpu"] if usa_gpu else CONFIG["batch_size_cpu"]
    inicio = time.perf_counter()
    saida = modelo.encode(
        textos,
        batch_size=batch,
        max_length=CONFIG["max_length"],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    emb = np.asarray(saida["dense_vecs"], dtype=np.float32)
    emb = normalize(emb, norm="l2").astype(np.float32)
    np.save(caminho_cache, emb)
    print(f"Embeddings salvos: {emb.shape}; tempo: {time.perf_counter()-inicio:.2f}s")
    return emb


def preparar_espaco_clusterizacao(x_dev, x_teste):
    n_comp = min(CONFIG["n_componentes_pca"], x_dev.shape[1], len(x_dev) - 1)
    pca = PCA(n_components=n_comp, random_state=CONFIG["seed"])
    z_dev = normalize(pca.fit_transform(x_dev)).astype(np.float32)
    z_teste = normalize(pca.transform(x_teste)).astype(np.float32)
    print(
        f"PCA ajustada apenas no desenvolvimento: {n_comp} dimensoes; "
        f"variancia={pca.explained_variance_ratio_.sum():.4f}"
    )
    return z_dev, z_teste


def silhouette_amostrado(x, labels, ignorar_ruido=False, selecao_k=False):
    labels = np.asarray(labels)
    mascara = labels != -1 if ignorar_ruido else np.ones(len(labels), dtype=bool)
    xv, lv = x[mascara], labels[mascara]
    if len(xv) < 3 or len(np.unique(lv)) < 2 or len(np.unique(lv)) >= len(xv):
        return np.nan
    limite = CONFIG["amostra_silhouette_k"] if selecao_k else CONFIG["amostra_silhouette_avaliacao"]
    n = min(limite, len(xv))
    return silhouette_score(
        xv, lv, metric="euclidean", sample_size=n, random_state=CONFIG["seed"]
    )


def selecionar_k_por_silhouette(x_dev, pastas):
    limite = min(CONFIG["k_max"], len(x_dev) - 1)
    linhas = []
    print("\nSelecionando K exclusivamente no desenvolvimento...")
    for k in range(CONFIG["k_min"], limite + 1, CONFIG["k_passo"]):
        inicio = time.perf_counter()
        modelo = MiniBatchKMeans(
            n_clusters=k, batch_size=2048, n_init=5,
            max_iter=200, random_state=CONFIG["seed"]
        )
        labels = modelo.fit_predict(x_dev)
        sil = silhouette_amostrado(x_dev, labels, selecao_k=True)
        linhas.append({"k": k, "silhouette": sil, "tempo_segundos": time.perf_counter()-inicio})
        print(f"K={k:>3} | silhouette={sil:.5f}")

    resultado = pd.DataFrame(linhas)
    melhor_k = int(resultado.loc[resultado["silhouette"].idxmax(), "k"])
    resultado.to_csv(
        pastas["tabelas"] / "selecao_k_desenvolvimento.csv",
        index=False, encoding="utf-8-sig"
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=resultado, x="k", y="silhouette", marker="o")
    plt.axvline(melhor_k, color="red", linestyle="--", label=f"K escolhido = {melhor_k}")
    plt.title("Escolha automatica de K pelo silhouette no desenvolvimento - Wiki")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / "selecao_k_silhouette.png", dpi=180)
    plt.close()
    print(f"K selecionado: {melhor_k}")
    return melhor_k


def purity_inverse_hmp(y_true, y_pred):
    matriz = contingency_matrix(y_true, y_pred)
    purity = matriz.max(axis=0).sum() / matriz.sum()
    inverse = matriz.max(axis=1).sum() / matriz.sum()
    hmp = 0 if purity + inverse == 0 else 2 * purity * inverse / (purity + inverse)
    return purity, inverse, hmp


def avaliar(nome, particao, x, y_true, labels, tempo, parametros=None):
    labels = np.asarray(labels)
    mascara = labels != -1
    n_clusters = len(set(labels)) - int(-1 in labels)
    cobertura = mascara.mean()
    linha = {
        "algoritmo": nome,
        "particao": particao,
        "n_clusters": n_clusters,
        "ruido_n": int((~mascara).sum()),
        "ruido_pct": 100 * (1 - cobertura),
        "cobertura_pct": 100 * cobertura,
        "tempo_segundos": tempo,
        "parametros": json.dumps(parametros or {}, ensure_ascii=False, default=str),
    }
    if mascara.sum() and n_clusters >= 1:
        yt, yp = np.asarray(y_true)[mascara], labels[mascara]
        purity, inverse, hmp = purity_inverse_hmp(yt, yp)
        linha.update({
            "NMI": normalized_mutual_info_score(yt, yp),
            "ARI": adjusted_rand_score(yt, yp),
            "Purity": purity,
            "Inverse_Purity": inverse,
            "HMP": hmp,
            "Silhouette": silhouette_amostrado(x, labels, ignorar_ruido=True),
        })
    else:
        linha.update({m: np.nan for m in [
            "NMI", "ARI", "Purity", "Inverse_Purity", "HMP", "Silhouette"
        ]})
    return linha


def testar_particionais(x_dev, x_teste, y_dev, y_teste, k):
    modelos = {
        "kmeans": KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=CONFIG["seed"]),
        "minibatch_kmeans": MiniBatchKMeans(
            n_clusters=k, batch_size=2048, n_init=10,
            max_iter=300, random_state=CONFIG["seed"]
        ),
        "birch": Birch(n_clusters=k, threshold=0.5, branching_factor=50),
    }
    metricas, predicoes = [], {}
    for nome in CONFIG["algoritmos_particionais"]:
        modelo = modelos[nome]
        inicio = time.perf_counter()
        modelo.fit(x_dev)
        tempo_fit = time.perf_counter() - inicio
        pred_dev = modelo.predict(x_dev)
        pred_teste = modelo.predict(x_teste)
        predicoes[nome] = {"desenvolvimento": pred_dev, "teste": pred_teste}
        metricas.append(avaliar(nome, "desenvolvimento", x_dev, y_dev, pred_dev, tempo_fit, {"k": k}))
        metricas.append(avaliar(nome, "teste", x_teste, y_teste, pred_teste, 0.0, {"k": k}))
    return metricas, predicoes


def ajustar_hdbscan_no_desenvolvimento(x_dev, y_dev, pastas):
    resultados = []
    melhor_modelo = None
    melhor_nome = None
    melhor_criterio = -np.inf
    melhor_tempo = 0.0
    print("\nAjustando HDBSCAN somente no desenvolvimento...")

    for tamanho in CONFIG["hdbscan_min_cluster_size"]:
        for amostras in CONFIG["hdbscan_min_samples"]:
            for metodo in CONFIG["hdbscan_selection_method"]:
                parametros = {
                    "min_cluster_size": tamanho,
                    "min_samples": amostras,
                    "cluster_selection_method": metodo,
                }
                nome = f"hdbscan_mcs{tamanho}_ms{amostras}_{metodo}"
                inicio = time.perf_counter()
                modelo = hdbscan.HDBSCAN(
                    metric="euclidean",
                    prediction_data=True,
                    core_dist_n_jobs=-1,
                    gen_min_span_tree=False,
                    **parametros,
                )
                labels = modelo.fit_predict(x_dev)
                tempo = time.perf_counter() - inicio
                linha = avaliar(nome, "desenvolvimento", x_dev, y_dev, labels, tempo, parametros)
                sil = linha["Silhouette"]
                cobertura = linha["cobertura_pct"] / 100
                criterio = sil * cobertura ** CONFIG["peso_cobertura_hdbscan"] if pd.notna(sil) else -np.inf
                linha["criterio_interno"] = criterio
                resultados.append(linha)
                print(
                    f"{nome} | clusters={linha['n_clusters']} | sil={sil:.4f} | "
                    f"cobertura={linha['cobertura_pct']:.2f}% | criterio={criterio:.4f}"
                )
                if linha["n_clusters"] > 1 and np.isfinite(criterio) and criterio > melhor_criterio:
                    melhor_criterio = criterio
                    melhor_modelo = modelo
                    melhor_nome = nome
                    melhor_tempo = tempo

    tabela = pd.DataFrame(resultados).sort_values("criterio_interno", ascending=False)
    tabela.to_csv(
        pastas["tabelas"] / "selecao_hdbscan_desenvolvimento.csv",
        index=False, encoding="utf-8-sig"
    )
    if melhor_modelo is None:
        raise RuntimeError("Nenhuma configuracao valida de HDBSCAN foi encontrada.")
    return melhor_nome, melhor_modelo, melhor_tempo


def testar_hdbscan(x_dev, x_teste, y_dev, y_teste, pastas):
    nome, modelo, tempo_fit = ajustar_hdbscan_no_desenvolvimento(x_dev, y_dev, pastas)
    labels_dev = modelo.labels_
    inicio = time.perf_counter()
    labels_teste, forca = approximate_predict(modelo, x_teste)
    tempo_pred = time.perf_counter() - inicio
    parametros = modelo.get_params(deep=False)
    metricas = [
        avaliar(nome, "desenvolvimento", x_dev, y_dev, labels_dev, tempo_fit, parametros),
        avaliar(nome, "teste", x_teste, y_teste, labels_teste, tempo_pred, parametros),
    ]
    return metricas, {nome: {"desenvolvimento": labels_dev, "teste": labels_teste}}, forca


def salvar_listas(df, labels, algoritmo, particao, pastas):
    temp = df[["document_id", "label_name", "text", "origem_original"]].copy()
    temp["cluster"] = labels
    temp.to_csv(
        pastas["tabelas"] / f"documentos_{algoritmo}_{particao}.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.crosstab(temp["cluster"], temp["label_name"]).to_csv(
        pastas["tabelas"] / f"contingencia_{algoritmo}_{particao}.csv",
        encoding="utf-8-sig"
    )
    with open(
        pastas["listas"] / f"clusters_{algoritmo}_{particao}.txt", "w", encoding="utf-8"
    ) as arq:
        for cluster, grupo in temp.groupby("cluster", sort=True):
            arq.write(f"\nCLUSTER {cluster} | {len(grupo)} documentos\n" + "=" * 90 + "\n")
            for label, qtd in grupo["label_name"].value_counts().head(CONFIG["top_labels_por_cluster"]).items():
                arq.write(f"{label}: {qtd} ({100*qtd/len(grupo):.2f}%)\n")
            for _, linha in grupo.head(CONFIG["top_documentos_por_cluster"]).iterrows():
                texto = " ".join(str(linha["text"]).split())[:500]
                arq.write(f"ID={linha['document_id']} | GT={linha['label_name']}\n{texto}\n")


def gerar_graficos(metricas, pastas):
    colunas = ["NMI", "ARI", "Purity", "Inverse_Purity", "HMP", "Silhouette", "cobertura_pct"]
    dados = metricas.melt(
        id_vars=["algoritmo", "particao"], value_vars=colunas,
        var_name="metrica", value_name="valor"
    )
    dados.loc[dados["metrica"] == "cobertura_pct", "valor"] /= 100
    grafico = sns.catplot(
        data=dados, x="algoritmo", y="valor", hue="metrica", col="particao",
        kind="bar", height=7, aspect=1.3, sharex=False, errorbar=None
    )
    grafico.set_xticklabels(rotation=45, ha="right")
    grafico.set(ylim=(-0.1, 1.05))
    grafico.figure.subplots_adjust(bottom=0.28)
    grafico.figure.savefig(pastas["graficos"] / "metricas_desenvolvimento_teste.png", dpi=180)
    plt.close(grafico.figure)

    teste = metricas[metricas["particao"] == "teste"]
    plt.figure(figsize=(14, 6))
    sns.scatterplot(data=teste, x="cobertura_pct", y="NMI", hue="algoritmo", s=140)
    plt.title("Compromisso entre cobertura e NMI no teste - Wiki")
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / "cobertura_vs_nmi_teste.png", dpi=180)
    plt.close()


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(CONFIG["seed"])
    pastas = criar_pastas()
    df = carregar_wiki()
    dev, teste = dividir_desenvolvimento_teste(df)
    dev.to_csv(pastas["tabelas"] / "particao_desenvolvimento.csv", index=False, encoding="utf-8-sig")
    teste.to_csv(pastas["tabelas"] / "particao_teste.csv", index=False, encoding="utf-8-sig")

    modelo_bge, usa_gpu = carregar_modelo_bgem3()
    x_dev = gerar_embeddings(
        modelo_bge, dev["text"].tolist(),
        pastas["cache"] / "embeddings_dev_wiki_bgem3.npy", usa_gpu
    )
    x_teste = gerar_embeddings(
        modelo_bge, teste["text"].tolist(),
        pastas["cache"] / "embeddings_teste_wiki_bgem3.npy", usa_gpu
    )
    del modelo_bge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    z_dev, z_teste = preparar_espaco_clusterizacao(x_dev, x_teste)
    melhor_k = selecionar_k_por_silhouette(z_dev, pastas)
    y_dev = dev["label_name"].astype(str).to_numpy()
    y_teste = teste["label_name"].astype(str).to_numpy()

    metricas_p, predicoes_p = testar_particionais(z_dev, z_teste, y_dev, y_teste, melhor_k)
    metricas_h, predicoes_h, forca_h = testar_hdbscan(z_dev, z_teste, y_dev, y_teste, pastas)
    metricas = pd.DataFrame(metricas_p + metricas_h)
    metricas.to_csv(pastas["tabelas"] / "comparacao_final.csv", index=False, encoding="utf-8-sig")

    predicoes = {**predicoes_p, **predicoes_h}
    for algoritmo, partes in predicoes.items():
        salvar_listas(dev, partes["desenvolvimento"], algoritmo, "desenvolvimento", pastas)
        salvar_listas(teste, partes["teste"], algoritmo, "teste", pastas)

    gerar_graficos(metricas, pastas)
    resumo = {
        "config": CONFIG,
        "melhor_k": melhor_k,
        "n_desenvolvimento": len(dev),
        "n_teste": len(teste),
        "observacao": (
            "K e hiperparametros foram escolhidos apenas no desenvolvimento; "
            "ground truth foi usado somente para avaliacao e estratificacao."
        ),
    }
    with open(pastas["raiz"] / "metadados.json", "w", encoding="utf-8") as arq:
        json.dump(resumo, arq, ensure_ascii=False, indent=2, default=str)

    print("\nRESULTADO FINAL")
    print(metricas.sort_values(["particao", "NMI"], ascending=[True, False]).to_string(index=False))
    print(f"\nArquivos salvos em: {pastas['raiz'].resolve()}")


if __name__ == "__main__":
    main()