import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def make_dataframe(contents):
    """
    Cria um DataFrame a partir de uma lista de dicionários.

    Args:
        contents (list): Uma lista de dicionários.

    Returns:
        pd.DataFrame: Um DataFrame contendo os dados dos dicionários.
    """
    if contents is None:
        return None

    contents_df = pd.DataFrame(contents)
    # Colunas a serem removidas
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

    # Substituir valores incompletos por NaN
    contents_df.replace("N/A", np.nan, inplace=True)
    contents_df.replace("None", np.nan, inplace=True)

    # Remover caracteres para converter em valores numéricos
    contents_df.rename(columns={"Runtime": "Runtime (min)"}, inplace=True)
    replace = {"Year": "-", "imdbVotes": ",", "imdbRating": "", "Runtime (min)": " min"}
    for key, value in replace.items():
        contents_df[key] = contents_df[key].str.replace(value, "", regex=False)

    # Converter para valores numéricos
    numeric_columns = ["imdbVotes", "imdbRating", "Metascore", "Runtime (min)"]
    for col in numeric_columns:
        contents_df[col] = pd.to_numeric(contents_df[col], errors="coerce")

    # Preencher valores NaN
    fill_values = {
        "Rated": "Not_Rated",
        "Metascore": 0,
        "Runtime (min)": 0,
        "imdbRating": 0,
        "imdbVotes": 0,
        "Year": 0,
    }
    contents_df.fillna(fill_values, inplace=True)
    contents_df["Rated"] = contents_df["Rated"].str.upper()
    contents_df["Rated"] = contents_df["Rated"].str.replace("NOT RATED", "NOT_RATED")
    int_columns = ["imdbVotes", "Metascore", "Runtime (min)"]
    for col in int_columns:
        contents_df[col] = contents_df[col].astype(int)

    # Remover caracteres não alfanuméricos
    chars = ",;:-()[]{}'\""
    for col in contents_df.columns:
        if contents_df[col].dtype == "object":
            for c in chars:
                contents_df[col] = contents_df[col].astype(str).str.replace(c, "")

    # Converter nomes de países para um único nome
    countries = {
        "United States": "USA",
        "United Kingdom": "UK",
        "United Arab Emirates": "UAE",
        "Republic of Ireland": "Ireland",
    }
    for key, value in countries.items():
        contents_df["Country"] = contents_df["Country"].str.replace(key, value)

    return contents_df


def get_features(contents_df, item_id):
    """
    Obtém as características de um item a partir de um DataFrame.

    Args:
        contents_df (pd.DataFrame): O DataFrame contendo as características dos itens.
        item_id (str): O ID do item.

    Returns:
        dict: Um dicionário com as características do item.
    """
    item = contents_df[contents_df["ItemId"] == item_id]
    if item.empty:
        return None

    # Separar as colunas categóricas
    category_columns = [
        col for col in contents_df.columns if contents_df[col].dtype == "object"
    ]
    category_columns.remove("ItemId")
    features = {}
    for col in category_columns:
        features[col] = item[col].values[0]

    # Converter valores NaN para strings vazias
    for key, value in features.items():
        if value == "nan" or pd.isna(value):
            features[key] = ""

    return features


def features_to_str(features):
    """
    Converte um dicionário de características em uma string.

    Args:
        features (dict): Um dicionário de características.

    Returns:
        str: Uma string de características.
    """
    if features is None:
        return ""

    text = []
    for _, value in features.items():
        text.append(f"{value}")

    return " ".join(text)


def get_tfidf(contents_df, max_features=10_000):
    """
    Calcula a Term Frequency-Inverse Document Frequency para as características de cada item no dataframe.

    Args:
        contents_df (pd.DataFrame): Dataframe contendo as características dos itens.
        max_features (int): Número máximo de características a serem consideradas.

    Returns:
        scipy.sparse.csr_matrix: Matriz TF-IDF.
    """
    print(f"Calculando TF-IDF com {max_features} features...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    l = len(contents_df)
    printProgressBar(0, l, length=50)
    item_features_text = []

    # Aplicar features_to_str para as características de cada item
    for i, row in contents_df.iterrows():
        item_id = row["ItemId"]
        features = get_features(contents_df, item_id)
        item_features_text.append(features_to_str(features))
        printProgressBar(i + 1, l, length=50)

    tfidf_matrix = vectorizer.fit_transform(item_features_text)
    print("TF-IDF Completo.")
    return tfidf_matrix


def get_item_vector(item_id, items_index, tfidf_matrix):
    """
    Obtém a representação vetorial de um item.

    Args:
        item_id (str): O ID do item.
        items_index (dict): Um dicionário que mapeia IDs de itens para seus índices na matriz TF-IDF.
        tfidf_matrix (scipy.sparse.csr_matrix): A matriz TF-IDF.

    Returns:
        numpy.ndarray: A representação vetorial do item.
    """
    if item_id not in items_index:
        return None
    item_index = items_index[item_id]
    item_vector = tfidf_matrix[item_index].toarray()[0]
    return item_vector


def get_user_vector(user_id, ratings_df, utility_matrix, tfidf_matrix):
    """
    Obtém a representação vetorial de um usuário.

    Args:
        user_id (str): O ID do usuário.
        ratings_df (pd.DataFrame): DataFrame contendo as avaliações de usuário-item.
        utility_matrix (scipy.sparse.csr_matrix): A matriz de notas de usuários-itens.
        tfidf_matrix (scipy.sparse.csr_matrix): A matriz TF-IDF.

    Returns:
        numpy.ndarray: A representação vetorial do usuário.
    """
    user_map = {user: i for i, user in enumerate(ratings_df["UserId"].unique())}
    u = utility_matrix[user_map[user_id], :].tocsr()
    user_vector = np.dot(u, tfidf_matrix).toarray()
    return user_vector


def cos_sim(v1, v2):
    """
    Calcula a similaridade do cosseno entre dois vetores.

    Args:
        v1 (numpy.ndarray): O primeiro vetor.
        v2 (numpy.ndarray): O segundo vetor.

    Returns:
        float: A similaridade do cosseno entre os dois vetores.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_utility_matrix(ratings_df, contents_df):
    """
    Cria uma matriz de utilidade a partir de um dataframe de avaliações.

    Args:
        ratings_df (pd.DataFrame): DataFrame contendo avaliações de usuário-item.
        contents_df (pd.DataFrame): Dataframe contendo características dos itens.

    Returns:
        scipy.sparse.csr_matrix: A matriz de utilidade.
    """
    # Criar mapeamentos de usuários e itens
    user_map = {user: i for i, user in enumerate(ratings_df["UserId"].unique())}
    item_map = {item: i for i, item in enumerate(contents_df["ItemId"])}

    # Extrair dados para matriz esparsa
    rows = [user_map[user] for user in ratings_df["UserId"]]
    cols = [item_map[item] for item in ratings_df["ItemId"]]
    data = ratings_df["Rating"]

    # Criar a matriz esparsa
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
    alpha=0.05,
    beta=0.5,
):
    """
    Obtém o ranking de itens para um usuário.

    Args:
        user_id (str): O ID do usuário.
        targets_df (pd.DataFrame): DataFrame contendo os alvos de usuário-item.
        contents_df (pd.DataFrame): Dataframe contendo as características dos itens.
        user_vector (numpy.ndarray): A representação vetorial do usuário.
        tfidf_matrix (scipy.sparse.csr_matrix): A matriz TF-IDF.
        item_map (dict): Mapeamento de itens para seus índices na matriz TF-IDF.
        top (int): O número de itens a serem retornados, Range(1-100).
        alpha (float): O peso para o metascore.
        beta (float): O peso para a classificação IMDB.

    Returns:
        dict: Um dicionário dos principais itens para o usuário.
    """
    # Encontrar os 100 itens do usuário
    if top < 1 or top > 100:
        raise ValueError("O parâmerto 'top' deve ser entre 1 e 100.")

    user_targets = targets_df.loc[targets_df["UserId"] == user_id, "ItemId"].tolist()
    item_data = contents_df.loc[
        contents_df["ItemId"].isin(user_targets),
        ["ItemId", "Metascore", "imdbRating", "imdbVotes"],
    ]
    r_ui = dict(zip(user_targets, np.zeros(len(user_targets))))

    # Fazer o cálculo de r_ui para cada item
    for item_id, metascore, imdb_rating, imdb_votes in zip(
        item_data["ItemId"],
        item_data["Metascore"],
        item_data["imdbRating"],
        item_data["imdbVotes"],
    ):
        item_vector = get_item_vector(item_id, item_map, tfidf_matrix)
        r_ui[item_id] = cos_sim(user_vector.flatten(), item_vector.flatten()) * (
            1 + alpha * metascore + beta * (imdb_rating + np.sqrt(imdb_votes))
        )

    # Ordernar os itens em ordem decrescente
    item_ranking = {
        k: v for k, v in sorted(r_ui.items(), key=lambda item: item[1], reverse=True)
    }

    top_items = dict(list(item_ranking.items())[:top])
    return top_items


def n_ratings(user_index, utility_matrix):
    """
    Obtém o número de avaliações de um usuário.

    Args:
        user_index (int): O índice do usuário.
        utility_matrix (scipy.sparse.csr_matrix): A matriz de notas de usuários-itens.

    Returns:
        int: O número de avaliações do usuário.
    """
    user_ratings = utility_matrix[user_index, :].nonzero()[1]
    return len(user_ratings)


def printProgressBar(
    iteration,
    total,
    prefix="Progreso",
    suffix="Completo",
    decimals=1,
    length=100,
    fill="█",
    end="\r",
):
    """
    Chame em um loop para criar uma barra de progresso no terminal

    Args:
        iteration (int): iteração atual
        total (int): total de iterações
        prefix (str): string de prefixo
        suffix (str): string de sufixo
        decimals (int): número positivo de decimais no percentual completo
        length (int): comprimento em caracteres da barra
        fill (str): caractere de preenchimento da barra
        end (str): caractere de fim (por exemplo, "\r", "\r\n")
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=end)
    # Imprimir nova linha ao completar
    if iteration == total:
        print()
