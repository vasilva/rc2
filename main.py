import pandas as pd
import json
from sys import argv
from aux import *


def main(n_features=50, alpha=0.1, beta=1.0):
    """
    Função principal para gerar rankings de usuários com base em avaliações, conteúdo e dados de alvo.
    Esta função executa os seguintes passos:
    1. Lê os arquivos de entrada (ratings.jsonl, content.jsonl e targets.csv).
    2. Cria uma matriz TF-IDF para os itens.
    3. Cria uma matriz de utilidade com base nas avaliações e no conteúdo.
    4. Gera rankings para cada usuário com base em suas avaliações e na matriz TF-IDF.
    5. Cria um DataFrame com os rankings de usuário-item.
    6. Exporta os rankings para um arquivo CSV para submissão.

    Uso:
    $ python3 main.py ratings.jsonl content.jsonl targets.csv
    """
    if len(argv) != 4:
        print("Uso:")
        print("python3 main.py ratings.jsonl content.jsonl targets.csv")
        return

    # Abrir ratings.jsonl
    with open(argv[1], "r") as f:
        ratings = [json.loads(line) for line in f]
    ratings_df = pd.DataFrame(ratings)

    # Abrir e editando content.jsonl
    with open(argv[2], "r") as f:
        contents = [json.loads(line) for line in f]
    contents_df = make_dataframe(contents)

    # Abrir targets.csv
    targets_df = pd.read_csv(argv[3])

    # Criar a matriz TF-IDF dos items
    tfidf_matrix = get_tfidf(contents_df, n_features)

    # Criar a matriz de notas
    utility_matrix = get_utility_matrix(ratings_df, contents_df)

    item_map = {item: i for i, item in enumerate(contents_df["ItemId"])}

    # Criar os rankings para cada usuário
    print("Calculando Rankings...")
    user_rankings = {}
    users_id = list(targets_df["UserId"].unique())
    l = len(users_id)
    printProgressBar(0, l, length=50)

    for i, user_id in enumerate(users_id):
        user_vector = get_user_vector(
            user_id,
            ratings_df,
            utility_matrix,
            tfidf_matrix,
        ).flatten()

        user_rankings[user_id] = get_item_ranking(
            user_id,
            targets_df,
            contents_df,
            user_vector,
            tfidf_matrix,
            item_map,
            alpha=alpha,
            beta=beta,
        )
        printProgressBar(i + 1, l, length=50)

    print("Rankings completos.")

    # Criar o Dataframe com os rankings
    user_item_pairs = [
        (user, item) for user, items in user_rankings.items() for item in items.keys()
    ]
    user_rankings_df = pd.DataFrame(user_item_pairs, columns=["UserId", "ItemId"])

    # Criar o arquivo csv para submissão
    user_rankings_df.to_csv(f"submission{n_features}_{alpha}_{beta}.csv", index=False)


if __name__ == "__main__":

    # Hiper-parâmetros com melhor valor
    n_features = 50
    alpha, beta = 0.5, 2.5
    main(n_features, alpha, beta)
