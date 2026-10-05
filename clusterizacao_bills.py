"""
Teste de clusterizacao no dataset completo Congressional Bills.

Objetivo
--------
1. Reutilizar exatamente o download e a leitura definidos em funcoes_rotina_final.py.
2. Unir bills_train e bills_test, remover textos duplicados e preservar o ground truth.
3. Representar os documentos com TF-IDF + TruncatedSVD, sem utilizar LLM.
4. Comparar algoritmos relativamente leves de clusterizacao.
5. Avaliar os clusters contra label_name com NMI, ARI, Purity, Inverse Purity e HMP.
6. Gerar tabelas, listas e graficos para auditoria dos resultados.

Observacao metodologica
-----------------------
O ground truth nao entra na vetorizacao. Por padrao, sua quantidade de classes e usada
para definir n_clusters nos algoritmos que exigem K. Isso permite uma comparacao direta,
mas deve ser declarado no experimento. Para eliminar essa informacao, altere
CONFIG["usar_qtd_ground_truth_como_k"] para False e informe CONFIG["k_fixo"].
"""

import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import (
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
)
try:
    from sklearn.cluster import BisectingKMeans
    BISECTING_DISPONIVEL = True
except ImportError:
    BISECTING_DISPONIVEL = False

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder, Normalizer

from funcoes_rotina_final import PreparadorDatasets


CONFIG = {
    "random_state": 42,
    "pasta_saida": "resultados_clusterizacao_bills",
    "max_features_tfidf": 30000,
    "min_df_tfidf": 3,
    "max_df_tfidf": 0.95,
    "ngram_range": (1, 2),
    "n_componentes_svd": 100,
    "usar_qtd_ground_truth_como_k": True,
    "k_fixo": 20,
    "algoritmos": [
        "kmeans",
        "minibatch_kmeans",
        "bisecting_kmeans",
        "birch",
        "dbscan",
    ],
    # DBSCAN e sensivel a escala. O codigo testa a grade abaixo e escolhe o maior
    # NMI apenas para diagnostico comparativo. Para avaliacao final, ajuste a grade
    # em uma base de desenvolvimento e congele os parametros antes do teste.
    "dbscan_eps": [0.35, 0.45, 0.55, 0.65, 0.75],
    "dbscan_min_samples": [5, 10],
    "amostra_silhouette": 5000,
    "amostra_visualizacao": 3000,
    "top_labels_por_cluster": 5,
    "top_documentos_por_cluster": 3,
    "max_clusters_heatmap": 40,
}


def criar_pastas():
    raiz = Path(CONFIG["pasta_saida"])
    pastas = {
        "raiz": raiz,
        "graficos": raiz / "graficos",
        "tabelas": raiz / "tabelas",
        "listas": raiz / "listas",
        "modelos": raiz / "modelos",
    }
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def carregar_bills_completo():
    """Reutiliza IDs, download, truncamento e deteccao de colunas do codigo original."""
    caminho_train = PreparadorDatasets._baixar_arquivo_drive("bills_train")
    caminho_test = PreparadorDatasets._baixar_arquivo_drive("bills_test")

    df_train = PreparadorDatasets._ler_jsonl_com_labels(caminho_train, "BILLS_TRAIN")
    df_test = PreparadorDatasets._ler_jsonl_com_labels(caminho_test, "BILLS_TEST")
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["origem"] = "train"
    df_test["origem"] = "test"

    df = pd.concat([df_train, df_test], ignore_index=True)
    n_antes = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    df["document_id"] = np.arange(len(df))

    if df.empty or df["label_name"].nunique() <= 1:
        raise ValueError("A base Bills nao possui documentos/labels suficientes.")

    print(f"Base completa: {len(df):,} documentos unicos")
    print(f"Duplicidades removidas: {n_antes - len(df):,}")
    print(f"Ground truths distintos: {df['label_name'].nunique():,}")
    return df


def vetorizar_textos(textos):
    print("\nVetorizando com TF-IDF...")
    inicio = time.perf_counter()
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        max_features=CONFIG["max_features_tfidf"],
        min_df=CONFIG["min_df_tfidf"],
        max_df=CONFIG["max_df_tfidf"],
        ngram_range=CONFIG["ngram_range"],
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_tfidf = tfidf.fit_transform(textos)

    limite = min(x_tfidf.shape[0] - 1, x_tfidf.shape[1] - 1)
    n_componentes = min(CONFIG["n_componentes_svd"], limite)
    if n_componentes < 2:
        raise ValueError("Numero insuficiente de dimensoes para executar o SVD.")

    svd = TruncatedSVD(n_components=n_componentes, random_state=CONFIG["random_state"])
    normalizador = Normalizer(copy=False)
    x_svd = normalizador.fit_transform(svd.fit_transform(x_tfidf)).astype(np.float32)
    tempo = time.perf_counter() - inicio

    print(f"TF-IDF: {x_tfidf.shape[0]:,} x {x_tfidf.shape[1]:,}")
    print(f"SVD: {x_svd.shape[0]:,} x {x_svd.shape[1]:,}")
    print(f"Variancia explicada acumulada: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Tempo da representacao: {tempo:.2f}s")
    return x_tfidf, x_svd, tfidf, svd, tempo


def purity_inverse_hmp(y_true, y_pred):
    matriz = contingency_matrix(y_true, y_pred)
    purity = np.sum(np.max(matriz, axis=0)) / np.sum(matriz)
    inverse = np.sum(np.max(matriz, axis=1)) / np.sum(matriz)
    hmp = 0.0 if purity + inverse == 0 else 2 * purity * inverse / (purity + inverse)
    return purity, inverse, hmp


def silhouette_seguro(x, clusters):
    mascara = clusters != -1
    x_valido = x[mascara]
    c_valido = clusters[mascara]
    if len(x_valido) < 3 or len(np.unique(c_valido)) < 2:
        return np.nan
    if len(np.unique(c_valido)) >= len(x_valido):
        return np.nan
    n = min(CONFIG["amostra_silhouette"], len(x_valido))
    return silhouette_score(
        x_valido,
        c_valido,
        metric="cosine",
        sample_size=n,
        random_state=CONFIG["random_state"],
    )


def avaliar_clusterizacao(nome, y_true, clusters, x, tempo):
    mascara = clusters != -1
    cobertura = float(mascara.mean())
    ruido = int((~mascara).sum())
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    if mascara.sum() == 0 or n_clusters < 1:
        return {
            "algoritmo": nome,
            "n_clusters": n_clusters,
            "ruido_n": ruido,
            "ruido_pct": 100 * (1 - cobertura),
            "cobertura_pct": 100 * cobertura,
            "NMI": np.nan,
            "ARI": np.nan,
            "Purity": np.nan,
            "Inverse_Purity": np.nan,
            "HMP": np.nan,
            "Silhouette_Cosine": np.nan,
            "tempo_segundos": tempo,
        }

    yt = y_true[mascara]
    yp = clusters[mascara]
    purity, inverse, hmp = purity_inverse_hmp(yt, yp)
    return {
        "algoritmo": nome,
        "n_clusters": n_clusters,
        "ruido_n": ruido,
        "ruido_pct": 100 * (1 - cobertura),
        "cobertura_pct": 100 * cobertura,
        "NMI": normalized_mutual_info_score(yt, yp),
        "ARI": adjusted_rand_score(yt, yp),
        "Purity": purity,
        "Inverse_Purity": inverse,
        "HMP": hmp,
        "Silhouette_Cosine": silhouette_seguro(x, clusters),
        "tempo_segundos": tempo,
    }


def executar_modelo(nome, modelo, x, y_true):
    print(f"Executando {nome}...")
    inicio = time.perf_counter()
    clusters = modelo.fit_predict(x)
    tempo = time.perf_counter() - inicio
    metrica = avaliar_clusterizacao(nome, y_true, clusters, x, tempo)
    print(
        f"  clusters={metrica['n_clusters']} | NMI={metrica['NMI']:.4f} | "
        f"ARI={metrica['ARI']:.4f} | HMP={metrica['HMP']:.4f} | {tempo:.2f}s"
    )
    return clusters, metrica


def executar_algoritmos(x, y_true, k):
    predicoes = {}
    metricas = []
    modelos = {
        "kmeans": KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=CONFIG["random_state"],
        ),
        "minibatch_kmeans": MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            n_init=5,
            max_iter=200,
            random_state=CONFIG["random_state"],
        ),
        "birch": Birch(n_clusters=k, threshold=0.5, branching_factor=50),
    }
    if BISECTING_DISPONIVEL:
        modelos["bisecting_kmeans"] = BisectingKMeans(
            n_clusters=k,
            n_init=3,
            random_state=CONFIG["random_state"],
        )

    for nome in CONFIG["algoritmos"]:
        if nome == "dbscan" or nome not in modelos:
            continue
        clusters, metrica = executar_modelo(nome, modelos[nome], x, y_true)
        predicoes[nome] = clusters
        metricas.append(metrica)

    if "dbscan" in CONFIG["algoritmos"]:
        resultados_dbscan = []
        for eps in CONFIG["dbscan_eps"]:
            for min_samples in CONFIG["dbscan_min_samples"]:
                nome = f"dbscan_eps{eps}_min{min_samples}"
                modelo = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean",
                    algorithm="auto",
                    n_jobs=-1,
                )
                clusters, metrica = executar_modelo(nome, modelo, x, y_true)
                predicoes[nome] = clusters
                metricas.append(metrica)
                resultados_dbscan.append(metrica)

    return predicoes, pd.DataFrame(metricas)


def criar_resumo_clusters(df, clusters, nome, pastas):
    temp = df[["document_id", "label_name", "text", "origem"]].copy()
    temp["cluster"] = clusters

    cruzada = pd.crosstab(temp["cluster"], temp["label_name"])
    cruzada.to_csv(pastas["tabelas"] / f"contingencia_{nome}.csv", encoding="utf-8-sig")

    linhas = []
    for cluster, grupo in temp.groupby("cluster", sort=True):
        contagens = grupo["label_name"].value_counts()
        total = len(grupo)
        top = contagens.head(CONFIG["top_labels_por_cluster"])
        for posicao, (label, quantidade) in enumerate(top.items(), start=1):
            linhas.append({
                "algoritmo": nome,
                "cluster": cluster,
                "tamanho_cluster": total,
                "posicao": posicao,
                "ground_truth": label,
                "quantidade": quantidade,
                "percentual_no_cluster": 100 * quantidade / total,
            })
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(
        pastas["tabelas"] / f"top_labels_por_cluster_{nome}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    arquivo_lista = pastas["listas"] / f"clusters_e_ground_truth_{nome}.txt"
    with open(arquivo_lista, "w", encoding="utf-8") as f:
        f.write(f"ALGORITMO: {nome}\n")
        f.write("=" * 100 + "\n")
        for cluster, grupo in temp.groupby("cluster", sort=True):
            f.write(f"\nCLUSTER {cluster} | {len(grupo)} documentos\n")
            f.write("-" * 100 + "\n")
            for label, quantidade in grupo["label_name"].value_counts().head(
                CONFIG["top_labels_por_cluster"]
            ).items():
                f.write(f"  {label}: {quantidade} ({100 * quantidade / len(grupo):.2f}%)\n")
            f.write("  Exemplos de documentos:\n")
            for _, linha in grupo.head(CONFIG["top_documentos_por_cluster"]).iterrows():
                texto = " ".join(str(linha["text"]).split())[:500]
                f.write(
                    f"    ID={linha['document_id']} | ground_truth={linha['label_name']} | "
                    f"origem={linha['origem']}\n    {texto}\n"
                )
    return temp, cruzada, resumo


def mapear_clusters_para_labels(df, clusters):
    """Mapeamento 1:1 pela atribuicao hungara, apenas para interpretacao."""

    clusters = np.asarray(clusters)

    # Remove os documentos classificados como ruido pelo DBSCAN.
    mascara_valida = clusters != -1

    if not mascara_valida.any():
        return pd.DataFrame(columns=[
            "cluster",
            "ground_truth_associado",
            "documentos_na_intersecao",
            "tamanho_cluster",
            "recall_da_classe",
            "precision_do_cluster",
        ])

    clusters_filtrados = clusters[mascara_valida]
    labels_filtrados = (
        df.loc[mascara_valida, "label_name"]
        .astype(str)
        .to_numpy()
    )

    # A tabela e construida somente com clusters validos.
    tabela = pd.crosstab(
        pd.Series(clusters_filtrados, name="cluster"),
        pd.Series(labels_filtrados, name="label_name"),
        dropna=False,
    )

    if tabela.empty:
        return pd.DataFrame(columns=[
            "cluster",
            "ground_truth_associado",
            "documentos_na_intersecao",
            "tamanho_cluster",
            "recall_da_classe",
            "precision_do_cluster",
        ])

    matriz = tabela.to_numpy()

    # Converte a maximizacao das intersecoes em problema de minimizacao.
    custo = matriz.max() - matriz
    linhas, colunas = linear_sum_assignment(custo)

    registros = []

    for i, j in zip(linhas, colunas):
        cluster = tabela.index[i]
        label = tabela.columns[j]
        documentos_intersecao = int(tabela.iloc[i, j])
        tamanho_cluster = int(tabela.iloc[i, :].sum())
        tamanho_classe = int(tabela.iloc[:, j].sum())

        registros.append({
            "cluster": cluster,
            "ground_truth_associado": label,
            "documentos_na_intersecao": documentos_intersecao,
            "tamanho_cluster": tamanho_cluster,
            "recall_da_classe": float(
                documentos_intersecao / max(1, tamanho_classe)
            ),
            "precision_do_cluster": float(
                documentos_intersecao / max(1, tamanho_cluster)
            ),
        })

    return (
        pd.DataFrame(registros)
        .sort_values("cluster")
        .reset_index(drop=True)
    )


def grafico_metricas(metricas, pastas):
    colunas = ["NMI", "ARI", "Purity", "Inverse_Purity", "HMP", "Silhouette_Cosine"]
    dados = metricas.melt(
        id_vars="algoritmo", value_vars=colunas, var_name="metrica", value_name="valor"
    )
    plt.figure(figsize=(15, 8))
    sns.barplot(data=dados, x="algoritmo", y="valor", hue="metrica")
    plt.ylim(-0.1, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.title("Comparacao dos algoritmos de clusterizacao - Bills")
    plt.xlabel("Algoritmo")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / "comparacao_metricas.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.barplot(data=metricas, x="algoritmo", y="tempo_segundos")
    plt.xticks(rotation=45, ha="right")
    plt.title("Tempo de processamento por algoritmo")
    plt.xlabel("Algoritmo")
    plt.ylabel("Segundos")
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / "comparacao_tempo.png", dpi=180)
    plt.close()


def grafico_heatmap(cruzada, nome, pastas):
    sem_ruido = cruzada.drop(index=-1, errors="ignore")
    if sem_ruido.empty:
        return
    top_clusters = sem_ruido.sum(axis=1).nlargest(CONFIG["max_clusters_heatmap"]).index
    top_labels = sem_ruido.sum(axis=0).nlargest(CONFIG["max_clusters_heatmap"]).index
    recorte = sem_ruido.loc[top_clusters, top_labels]
    normalizada = recorte.div(recorte.sum(axis=1).replace(0, np.nan), axis=0)

    plt.figure(figsize=(max(12, len(top_labels) * 0.45), max(7, len(top_clusters) * 0.35)))
    sns.heatmap(normalizada, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Distribuicao normalizada de ground truths por cluster - {nome}")
    plt.xlabel("Ground truth")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / f"heatmap_{nome}.png", dpi=180)
    plt.close()


def grafico_projecao(x, df, clusters, nome, pastas):
    rng = np.random.default_rng(CONFIG["random_state"])
    n = min(CONFIG["amostra_visualizacao"], len(df))
    indices = np.sort(rng.choice(len(df), size=n, replace=False))
    x_amostra = x[indices]
    clusters_amostra = clusters[indices]
    labels_amostra = df.iloc[indices]["label_name"].to_numpy()

    perplexidade = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexidade,
        init="pca",
        learning_rate="auto",
        random_state=CONFIG["random_state"],
    )
    coords = tsne.fit_transform(x_amostra)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1], hue=clusters_amostra,
        palette="tab20", s=16, linewidth=0, legend=False, ax=axes[0]
    )
    axes[0].set_title(f"Clusters encontrados - {nome}")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    codigos_label = LabelEncoder().fit_transform(labels_amostra)
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1], hue=codigos_label,
        palette="tab20", s=16, linewidth=0, legend=False, ax=axes[1]
    )
    axes[1].set_title("Ground truth dos mesmos documentos")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(pastas["graficos"] / f"projecao_tsne_{nome}.png", dpi=180)
    plt.close()


def salvar_metadados(df, metricas, tempo_representacao, k, pastas):
    dados = {
        "configuracao": CONFIG,
        "documentos_unicos": int(len(df)),
        "ground_truths_distintos": int(df["label_name"].nunique()),
        "k_utilizado": int(k),
        "tempo_representacao_segundos": float(tempo_representacao),
        "bibliotecas": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "melhor_por_nmi": (
            metricas.sort_values("NMI", ascending=False).iloc[0]["algoritmo"]
            if metricas["NMI"].notna().any() else None
        ),
    }
    with open(pastas["raiz"] / "metadados_execucao.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    pastas = criar_pastas()
    df = carregar_bills_completo()
    y_true = df["label_name"].astype(str).to_numpy()

    if CONFIG["usar_qtd_ground_truth_como_k"]:
        k = int(df["label_name"].nunique())
    else:
        k = int(CONFIG["k_fixo"])
    print(f"Quantidade K usada nos algoritmos particionais: {k}")

    _, x_svd, _, _, tempo_representacao = vetorizar_textos(df["text"].tolist())
    predicoes, metricas = executar_algoritmos(x_svd, y_true, k)
    metricas = metricas.sort_values(["NMI", "HMP"], ascending=False).reset_index(drop=True)
    metricas.to_csv(
        pastas["tabelas"] / "comparacao_algoritmos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_saida = df.copy()
    for nome, clusters in predicoes.items():
        df_saida[f"cluster_{nome}"] = clusters
    df_saida.to_csv(
        pastas["tabelas"] / "bills_documentos_com_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grafico_metricas(metricas, pastas)

    nomes_validos = metricas.loc[
        metricas["NMI"].notna() & (metricas["n_clusters"] > 1), "algoritmo"
    ].tolist()
    if not nomes_validos:
        raise RuntimeError("Nenhum algoritmo produziu clusterizacao valida.")

    # Relatorios completos para todos; graficos detalhados para o melhor por NMI.
    cruzadas = {}
    for nome in nomes_validos:
        _, cruzada, _ = criar_resumo_clusters(df, predicoes[nome], nome, pastas)
        cruzadas[nome] = cruzada
        mapa = mapear_clusters_para_labels(df, predicoes[nome])
        mapa.to_csv(
            pastas["tabelas"] / f"mapeamento_hungaro_{nome}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    melhor_nome = nomes_validos[0]
    grafico_heatmap(cruzadas[melhor_nome], melhor_nome, pastas)
    grafico_projecao(x_svd, df, predicoes[melhor_nome], melhor_nome, pastas)
    salvar_metadados(df, metricas, tempo_representacao, k, pastas)

    print("\n" + "=" * 90)
    print("RESULTADO FINAL")
    print("=" * 90)
    print(metricas.to_string(index=False))
    print(f"\nMelhor algoritmo por NMI: {melhor_nome}")
    print(f"Arquivos salvos em: {pastas['raiz'].resolve()}")


if __name__ == "__main__":
    main()