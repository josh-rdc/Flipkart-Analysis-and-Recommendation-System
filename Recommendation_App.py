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

flipkart_data = r"Dataset\flipkart_com-products.csv"
flipkart_data_df3_path = r"flipkart_data_df3.csv"

flipkart_data_df = import_csv_dataset(flipkart_data)
flipkart_data_df3 = import_csv_dataset(flipkart_data_df3_path)

flipkart_id_name = flipkart_data_df[['pid', 'product_name']]
flipkart_id_image = flipkart_data_df[['pid', 'image']]

# Load cosine similarity matrix
@st.cache_resource
def load_similarity_matrix():
    folder_path = "RecommendationFile"

    # Load all parts
    num_parts = 50
    cosine_sim_parts = []

    for i in range(num_parts):
        file_path = os.path.join(folder_path, f'cosine_sim_part_{i}.pkl')
        with open(file_path, 'rb') as f:
            cosine_sim_parts.append(pickle.load(f))

    # Merge into a single array
    cosine_sim = np.vstack(cosine_sim_parts)

    # print("Cosine similarity matrix successfully reconstructed.")

    return cosine_sim

cosine_sim = load_similarity_matrix()

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
def get_product_images(pid):
    image_row = flipkart_id_image[flipkart_id_image['pid'] == pid]['image']
    if not image_row.empty:
        images = ast.literal_eval(image_row.values[0])  # Convert string list to actual list
        return images
    return []

# Get Top 5 Similar Products
def get_top_5_similar(index, df, df_name, cosine_sim):
    pid = df.loc[index, 'pid']
    product_name = df_name.loc[df_name['pid'] == pid, 'product_name'].values[0]

    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    top_indices = [i[0] for i in sim_scores]

    top_pids = df.loc[top_indices, 'pid'].values
    top_names = df_name.loc[df_name['pid'].isin(top_pids), 'product_name'].values

    return pid, product_name, top_pids, top_names

# Streamlit UI
st.title("🛒 Flipkart Product Recommendation System")

# Dropdown to select product
selected_product = st.selectbox(
    "Select a product:",
    flipkart_id_name["product_name"].values
)

# Get the product details and index
if selected_product:
    selected_pid = flipkart_id_name[flipkart_id_name["product_name"] == selected_product]["pid"].values[0]
    selected_index = flipkart_data_df3[flipkart_data_df3["pid"] == selected_pid].index[0]

    # Get details
    product_details = get_product_details(selected_pid, flipkart_data_df3)

    # Display Product Details
    st.subheader(f"🎯 Selected Product: {selected_product}")
    st.write(f"**Brand:** {product_details['brand']}")
    st.write(f"**Category:** {product_details['category']}")
    st.write(f"**Retail Price:** {product_details['retail_price']}")
    st.write(f"**Discounted Price:** {product_details['discounted_price']}")
    st.write(f"**Description:** {product_details['description']}")

    # Display Product Images in a Single Row (from URLs)
    images = get_product_images(selected_pid)

    if images:
        cols = st.columns(len(images))  # Create dynamic columns based on the number of images
        for col, img_url in zip(cols, images):
            col.image(img_url, width=100)
    else:
        st.write("No images available.")

    # Get Recommendations
    pid, product_name, top_pids, top_names = get_top_5_similar(selected_index, flipkart_data_df3, flipkart_id_name, cosine_sim)

    # Display Recommended Products
    st.subheader("🔍 Top 5 Similar Products")
    for i, (top_pid, top_name) in enumerate(zip(top_pids, top_names), start=1):
        st.markdown(f"### {i}. {top_name}")

        # Get details
        top_details = get_product_details(top_pid, flipkart_data_df3)
        st.write(f"**Brand:** {top_details['brand']}")
        st.write(f"**Category:** {top_details['category']}")
        st.write(f"**Retail Price:** {top_details['retail_price']}")
        st.write(f"**Discounted Price:** {top_details['discounted_price']}")
        st.write(f"**Description:** {top_details['description']}")

        # Display Product Images in a Single Row (from URLs)
        top_images = get_product_images(top_pid)

        if top_images:
            cols = st.columns(len(top_images))  # Create dynamic columns based on number of images
            for col, img_url in zip(cols, top_images):
                col.image(img_url, width=100)
        else:
            st.write("No images available.")
