import pandas as pd
import json
from zipfile import ZipFile
from aux import *

if __name__ == "__main__":

    # Abrindo targets.csv
    with ZipFile("targets.zip", "r") as zip_targets:
        zip_targets.extract("targets.csv")

    targets_df = pd.read_csv("targets.csv")

    # Abrindo ratings.jsonl
    with ZipFile("ratings.zip", "r") as zip_ratings:
        zip_ratings.extract("ratings.jsonl")

    with open("ratings.jsonl", "r") as f:
        ratings = [json.loads(line) for line in f]

    ratings_df = pd.DataFrame(ratings)

    # Abrindo e editando content.jsonl
    with ZipFile("content.zip", "r") as zip_content:
        zip_content.extract("content.jsonl")

    with open("content.jsonl", "r") as f:
        contents = [json.loads(line) for line in f]

    contents_df = make_dataframe(contents)

    # Cria a matriz TF-IDF dos items
    tfidf_matrix, feature_names = get_tfidf(contents_df)

    # Criando a matriz de notas
    utility_matrix = get_utility_matrix(ratings_df, contents_df)

    item_map = {item: i for i, item in enumerate(contents_df["ItemId"])}

    # Criando os rankings para cada usuário
    print("Calculating rankings...")
    user_rankings = {}
    users_id = list(targets_df["UserId"].unique())
    l = len(users_id)
    printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)

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
        printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)

    print("Rankings complete.")

    # Criando o Dataframe com os rankings
    user_item_pairs = [
        (user, item) for user, items in user_rankings.items() for item in items.keys()
    ]
    user_rankings_df = pd.DataFrame(user_item_pairs, columns=["UserId", "ItemId"])

    # Criando o arquivo csv para submissão
    user_rankings_df.to_csv("submission.csv", index=False)
