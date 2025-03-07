import streamlit as st
import pandas as pd
import pickle
import ast
import os
import numpy as np

# Load DataFrames
@st.cache_data
def import_csv_dataset(file_path):
    df = pd.read_csv(file_path, header=0, encoding='latin-1')
    return df


# Load cosine similarity matrix
# @st.cache_resource
# def load_similarity_matrix():
#     folder_path = "RecommendationFile"

#     # Load all parts
#     num_parts = 50
#     cosine_sim_parts = []

#     for i in range(num_parts):
#         file_path = os.path.join(folder_path, f'cosine_sim_part_{i}.pkl')
#         with open(file_path, 'rb') as f:
#             cosine_sim_parts.append(pickle.load(f))

#     # Merge into a single array
#     cosine_sim = np.vstack(cosine_sim_parts)

#     # print("Cosine similarity matrix successfully reconstructed.")

#     return cosine_sim
# Load only the necessary part of the similarity matrix
@st.cache_data
def load_similarity_part(index):
    folder_path = "RecommendationFile"
    rows_per_file = 400  # Since 20,000 rows are split into 50 files

    # Determine which file contains the selected index
    file_number = index // rows_per_file
    file_path = os.path.join(folder_path, f'cosine_sim_part_{file_number}.pkl')

    # Load the specific file
    with open(file_path, 'rb') as f:
        cosine_sim_part = pickle.load(f)

    # Determine the row index within this part
    row_within_part = index % rows_per_file  

    return cosine_sim_part[row_within_part]

# Function to get product details
def get_product_details(pid, df):
    product = df[df['pid'] == pid].iloc[0]
    return {
        "brand": product['brand'],
        "category": product['category_0'],
        "retail_price": product['retail_price'],
        "discounted_price": product['discounted_price'],
        "description": product['description']
    }

# Function to get image URLs from flipkart_id_image dataframe
def get_product_images(pid, df):
    image_row = df[df['pid'] == pid]['image']
    if not image_row.empty:
        images = ast.literal_eval(image_row.values[0])  # Convert string list to actual list
        return images
    return []

# # Get Top 5 Similar Products
# def get_top_5_similar(index, df, df_name, cosine_sim):
#     pid = df.loc[index, 'pid']
#     product_name = df_name.loc[df_name['pid'] == pid, 'product_name'].values[0]

#     sim_scores = list(enumerate(cosine_sim[index]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
#     top_indices = [i[0] for i in sim_scores]

#     top_pids = df.loc[top_indices, 'pid'].values
#     top_names = df_name.loc[df_name['pid'].isin(top_pids), 'product_name'].values

#     return pid, product_name, top_pids, top_names

# Get Top 5 Similar Products (Updated)
def get_top_5_similar(index, df, df_name):
    pid = df.loc[index, 'pid']
    product_name = df_name.loc[df_name['pid'] == pid, 'product_name'].values[0]

    # Load only the relevant row from similarity matrix
    sim_scores = list(enumerate(load_similarity_part(index)))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 similar

    top_indices = [i[0] for i in sim_scores]
    top_pids = df.loc[top_indices, 'pid'].values
    top_names = df_name.loc[df_name['pid'].isin(top_pids), 'product_name'].values

    return pid, product_name, top_pids, top_names


import requests
from PIL import Image
from io import BytesIO

def fetch_image(img_url):
    try:
        response = requests.get(img_url, timeout=5)
        response.raise_for_status()  # Raise error for failed requests
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.write(f"Image failed to load: {e}")
        return None