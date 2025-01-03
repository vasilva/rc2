import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def make_dataframe(contents):
    """
    Create a DataFrame from a list of dictionaries.

    Args:
        contents (list): A list of dictionaries.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the dictionaries.
    """
    if contents is None:
        return None

    contents_df = pd.DataFrame(contents)
    # Columns to drop
    drop = [
        "Production",
        "Website",
        "Response",
        "totalSeasons",
        "Season",
        "Episode",
        "seriesID",
        "Poster",
        "Released",
        "Ratings",
        "Awards",
        "DVD",
        "BoxOffice",
    ]
    contents_df.drop(columns=drop, inplace=True)

    # Replacing incomplete values to NaN
    contents_df.replace("N/A", np.nan, inplace=True)
    contents_df.replace("None", np.nan, inplace=True)

    # Removing characters to convert to numeric values
    contents_df.rename(columns={"Runtime": "Runtime (min)"}, inplace=True)
    replace = {"Year": "-", "imdbVotes": ",", "imdbRating": "", "Runtime (min)": " min"}
    for key, value in replace.items():
        contents_df[key] = contents_df[key].str.replace(value, "", regex=False)

    # Converting to numeric
    numeric_columns = ["imdbVotes", "imdbRating", "Metascore", "Runtime (min)"]
    for col in numeric_columns:
        contents_df[col] = pd.to_numeric(contents_df[col], errors="coerce")

    # Filling NaN values
    fill_values = {
        "Rated": "Not_Rated",
        "Metascore": 0,
        "Runtime (min)": 0,
        "imdbRating": 0,
        "imdbVotes": 1,
        "Year": 0,
    }
    contents_df.fillna(fill_values, inplace=True)
    contents_df["Rated"] = contents_df["Rated"].str.upper()
    contents_df["Rated"] = contents_df["Rated"].str.replace("NOT RATED", "NOT_RATED")
    int_columns = ["imdbVotes", "Metascore", "Runtime (min)"]
    for col in int_columns:
        contents_df[col] = contents_df[col].astype(int)

    # Removing non-alphanumrics characters
    chars = ",;:-()[]{}'\""
    for col in contents_df.columns:
        if contents_df[col].dtype == "object":
            for c in chars:
                contents_df[col] = contents_df[col].astype(str).str.replace(c, "")

    # Converting country names to a single name
    countries = {
        "United States": "USA",
        "United Kingdom": "UK",
        "United Arab Emirates": "UAE",
        "Republic of Ireland": "Ireland",
        "South Korea": "Korea",
        "People's Republic of China": "China",
    }
    for key, value in countries.items():
        contents_df["Country"] = contents_df["Country"].str.replace(key, value)

    return contents_df


def get_features(contents_df, ItemId):
    """
    Get the features of an item from a DataFrame.

    Args:
        contents_df (pd.DataFrame): The DataFrame containing the item features.
        ItemId (str): The ID of the item.

    Returns:
        dict: A dictionary of the item's features.
    """
    item = contents_df[contents_df["ItemId"] == ItemId]
    if item.empty:
        return None

    category_columns = [
        col for col in contents_df.columns if contents_df[col].dtype == "object"
    ]
    category_columns.remove("ItemId")
    features = {}

    for col in category_columns:
        features[col] = item[col].values[0]

    # Convert NaN values to empty strings
    for key, value in features.items():
        if value == "nan" or pd.isna(value):
            features[key] = ""

    return features


def features_to_str(features):
    """
    Convert a dictionary of features to a string.

    Args:
        features (dict): A dictionary of features.

    Returns:
        str: A string of features.
    """
    if features is None:
        return ""

    text = []
    for _, value in features.items():
        text.append(f"{value}")

    return " ".join(text)


def get_tfidf(contents_df, max_features=100_000):
    """
    Calculates the Term Frequency-Inverse Document Frequency for the features of each item in the dataframe.

    Args:
        contents_df (pd.DataFrame): Dataframe containing item features.

    Returns:
        scipy.sparse.csr_matrix: TF-IDF matrix.
        list: Feature names.
    """
    print("Calculating TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    l = len(contents_df)
    printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
    item_features_text = []

    # Apply features_to_str to each item's features
    for i, row in contents_df.iterrows():
        item_id = row["ItemId"]
        features = get_features(contents_df, item_id)
        item_features_text.append(features_to_str(features))
        printProgressBar(i + 1, l, prefix="Progress:", suffix="Complete", length=50)

    tfidf_matrix = vectorizer.fit_transform(item_features_text)
    feature_names = vectorizer.get_feature_names_out()
    print("TF-IDF Complete.")
    return tfidf_matrix, feature_names


def get_item_vector(item_id, items_idx, tfidf_matrix):
    """
    Get the vector representation of an item.

    Args:
        item_id (str): The ID of the item.
        items_idx (dict): A dictionary mapping item IDs to their index in the tfidf matrix.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.

    Returns:
        numpy.ndarray: The vector representation of the item.
    """
    if item_id not in items_idx:
        return None
    item_index = items_idx[item_id]
    item_vector = tfidf_matrix[item_index].toarray()[0]
    return item_vector


def get_user_vector(user_id, ratings_df, utility_matrix, tfidf_matrix):
    """
    Get the vector representation of a user.

    Args:
        user_id (str): The ID of the user.
        ratings_df (pd.DataFrame): DataFrame containing user-item ratings.
        items_idx (dict): A dictionary mapping item IDs to their index in the tfidf matrix.
        utility_matrix (scipy.sparse.csr_matrix): The utility matrix.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.

    Returns:
        numpy.ndarray: The vector representation of the user.
    """
    user_map = {user: i for i, user in enumerate(ratings_df["UserId"].unique())}
    u = utility_matrix[user_map[user_id], :].tocsr()
    user_vector = np.dot(u, tfidf_matrix).toarray()
    return user_vector


def cos_sim(v1, v2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        v1 (numpy.ndarray): The first vector.
        v2 (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_utility_matrix(ratings_df, contents_df):
    """
    Create a utility matrix from a ratings dataframe.

    Args:
        ratings_df (pd.DataFrame): DataFrame containing user-item ratings.
        contents_df (pd.DataFrame): Dataframe containing item features.

    Returns:
        scipy.sparse.csr_matrix: The utility matrix.
    """
    # Create user and item mappings
    user_map = {user: i for i, user in enumerate(ratings_df["UserId"].unique())}
    item_map = {item: i for i, item in enumerate(contents_df["ItemId"])}

    # Extract data for sparse matrix
    rows = [user_map[user] for user in ratings_df["UserId"]]
    cols = [item_map[item] for item in ratings_df["ItemId"]]
    data = ratings_df["Rating"]

    # Create the sparse matrix
    sparse_matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(user_map), len(item_map)), dtype=np.int8
    )
    return sparse_matrix


def get_item_ranking(
    user_id,
    targets_df,
    contents_df,
    user_vector,
    tfidf_matrix,
    item_map,
    top=100,
    alpha=0.01,
    beta=0.1,
):
    """
    Get the ranking of items for a user.

    Args:
        user_id (str): The ID of the user.
        targets_df (pd.DataFrame): DataFrame containing user-item targets.
        contents_df (pd.DataFrame): Dataframe containing item features.
        user_vector (numpy.ndarray): The vector representation of the user.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix.
        top (int): The number of top items to consider.
        alpha (float): The weight for the metascore.
        beta (float): The weight for the IMDB rating.

    Returns:
        dict: A dictionary of the top items for the user.
    """
    user_targets = targets_df.loc[targets_df["UserId"] == user_id, "ItemId"].tolist()
    item_data = contents_df.loc[
        contents_df["ItemId"].isin(user_targets),
        ["ItemId", "Metascore", "imdbRating", "imdbVotes"],
    ]
    r_ui = dict(zip(user_targets, np.zeros(len(user_targets))))

    for item_id, metascore, imdb_rating, imdb_votes in zip(
        item_data["ItemId"],
        item_data["Metascore"],
        item_data["imdbRating"],
        item_data["imdbVotes"],
    ):
        item_vector = get_item_vector(item_id, item_map, tfidf_matrix)
        r_ui[item_id] = cos_sim(user_vector.flatten(), item_vector.flatten()) * (
            1
            + alpha * metascore
            + beta * (imdb_rating - imdb_rating / (1 + np.log(imdb_votes)))
        )

    item_ranking = {
        k: v for k, v in sorted(r_ui.items(), key=lambda item: item[1], reverse=True)
    }
    top_items = dict(list(item_ranking.items())[:top])
    return top_items


def n_ratings(user_index, utility_matrix):
    """
    Get the number of ratings for a user.

    Args:
        user_index (int): The index of the user.
        utility_matrix (scipy.sparse.csr_matrix): The utility matrix.

    Returns:
        int: The number of ratings for the user.
    """
    user_ratings = utility_matrix[user_index, :].nonzero()[1]
    return len(user_ratings)


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration (int): current iteration
        total (int): total iterations
        prefix (str): prefix string
        suffix (str): suffix string
        decimals (int): positive number of decimals in percent complete
        length (int): character length of bar
        fill (str): bar fill character
        printEnd (str): end character (e.g. "\r", "\r\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
