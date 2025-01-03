import pandas as pd
import json
from sys import argv
from aux import *

if __name__ == "__main__":

    n = len(argv)
    if n == 1:
        argv.append("ratings.jsonl")
        n = len(argv)
    if n == 2:
        argv.append("content.jsonl")
        n = len(argv)
    if n == 3:
        argv.append("targets.csv")
        n = len(argv)

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
    tfidf_matrix, feature_names = get_tfidf(contents_df)

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
            user_id, ratings_df, utility_matrix, tfidf_matrix
        ).flatten()
        user_rankings[user_id] = get_item_ranking(
            user_id,
            targets_df,
            contents_df,
            user_vector,
            tfidf_matrix,
            item_map,
            alpha=0.1,
            beta=1.0,
        )
        printProgressBar(i + 1, l, length=50)

    print("Rankings completos.")

    # Criar o Dataframe com os rankings
    user_item_pairs = [
        (user, item) for user, items in user_rankings.items() for item in items.keys()
    ]
    user_rankings_df = pd.DataFrame(user_item_pairs, columns=["UserId", "ItemId"])

    # Criar o arquivo csv para submissão
    user_rankings_df.to_csv("submission.csv", index=False)
