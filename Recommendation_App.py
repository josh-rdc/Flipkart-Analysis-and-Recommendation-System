import streamlit as st
import pandas as pd
import pickle
import ast
import os
import numpy as np

from Utilities.Functions import import_csv_dataset
# from Utilities.Functions import load_similarity_matrix
from Utilities.Functions import get_product_details
from Utilities.Functions import get_product_images
from Utilities.Functions import get_top_5_similar

from Utilities.Functions import fetch_image

# Load DataFrames
flipkart_data = "Dataset/flipkart_com-products.csv"
flipkart_data_df3_path = "flipkart_data_df3.csv"

flipkart_data_df = import_csv_dataset(flipkart_data)
flipkart_data_df3 = import_csv_dataset(flipkart_data_df3_path)

flipkart_id_name = flipkart_data_df[['pid', 'product_name']]
flipkart_id_image = flipkart_data_df[['pid', 'image']]

# Add flipkart_id_name['product_name'] to flipkart_data_df3
flipkart_data_df3 = pd.merge(flipkart_data_df3, flipkart_id_name, on='pid')

st.title("🛒 Flipkart Product Recommendation System")

# Load Cosine Similarity Matrix
# cosine_sim = load_similarity_matrix()

# Sidebar Taskbar for Product Selection
# Ensure missing values are handled properly
flipkart_data_df3.fillna("", inplace=True)  # Replace NaN with empty strings

# Select Main Category 
# Count frequency of each category
category_counts = flipkart_data_df3["category_0"].value_counts()

# Get top 30 categories based on frequency
top_30_categories = sorted(category_counts.head(30).index)  # Sort alphabetically

# Get remaining categories
remaining_categories = sorted(category_counts.index.difference(top_30_categories))  # Sort alphabetically

# Combine both lists
sorted_categories = top_30_categories + remaining_categories  

# Create dropdown with sorted categories
selected_main_category = st.sidebar.selectbox("Select Main Category:", sorted_categories)


# Filter for Category 1 (Allow Empty Option)
filtered_df1 = flipkart_data_df3[flipkart_data_df3["category_0"] == selected_main_category]
categories_1 = sorted(filtered_df1["category_1"].dropna().unique())  
categories_1.insert(0, "All")  # Add an "All" option
selected_category_1 = st.sidebar.selectbox("Select Sub-Category 1:", categories_1)

# Apply filtering for Category 1
if selected_category_1 != "All":
    filtered_df1 = filtered_df1[filtered_df1["category_1"] == selected_category_1]

# Filter for Category 2 (Allow Empty Option)
categories_2 = sorted(filtered_df1["category_2"].dropna().unique())  
categories_2.insert(0, "All")  # Add an "All" option
selected_category_2 = st.sidebar.selectbox("Select Sub-Category 2:", categories_2)

# Apply filtering for Category 2
if selected_category_2 != "All":
    filtered_df1 = filtered_df1[filtered_df1["category_2"] == selected_category_2]

# Filter and Sort Products
sorted_products = sorted(filtered_df1["product_name"].values)  
selected_product = st.sidebar.selectbox("Select a Product:", sorted_products)


# Get the product details and index
if selected_product:
    selected_pid = flipkart_id_name[flipkart_id_name["product_name"] == selected_product]["pid"].values[0]
    selected_index = flipkart_data_df3[flipkart_data_df3["pid"] == selected_pid].index[0]

    # Get details
    product_details = get_product_details(selected_pid, flipkart_data_df3)
    
    # Get Recommendations
    # pid, product_name, top_pids, top_names = get_top_5_similar(selected_index, flipkart_data_df3, flipkart_id_name)
    # pid, product_name, top_pids, top_names = get_top_5_similar(selected_index, flipkart_data_df3, flipkart_id_name, cosine_sim)
 
    col1, col2 = st.columns(2)

    # Left Column: Selected Product
    with col1:
        with st.container(border=True,height=900):
            st.subheader(f"🖱️ Selected Product")
            st.markdown(f"### {selected_product}")
            
            # Create a DataFrame for better formatting
            selected_product_df = pd.DataFrame({
                "Attribute": ["Brand", "Category", "Retail Price", "Discounted Price", "Description"],
                "Details": [
                    str(product_details['brand']),
                    str(product_details['category']),
                    str(product_details['retail_price']),
                    str(product_details['discounted_price']),
                    str(product_details['description'])
                ]
            })

            
            # Ensure all values are strings
            selected_product_df["Attribute"] = selected_product_df["Attribute"].astype(str)
            selected_product_df["Details"] = selected_product_df["Details"].astype(str)

            # Display as a table
            st.table(selected_product_df)

            # Display Product Images
            images = get_product_images(selected_pid, flipkart_id_image)

            # Fetch and display images
            img_objects = [fetch_image(img_url) for img_url in images]

            img_cols = st.columns(len(img_objects))
            for col, img_obj in zip(img_cols, img_objects):
                if img_obj:
                    col.image(img_obj, width=100)
                else:
                    col.write("Image not available")

            # if images:
            #     img_cols = st.columns(len(images))  
            #     for col, img_url in zip(img_cols, images):
            #         col.image(img_url, width=100)
            # else:
            #     st.write("No images available.")

    # Right Column: Top 5 Similar Products
    with col2:
        with st.container(border=True, height=900):
            st.subheader("🔍 Top 5 Similar Products")

            # Get recommendations
            # pid, product_name, top_pids, top_names = get_top_5_similar(selected_index, flipkart_data_df3, flipkart_id_name, cosine_sim)
            pid, product_name, top_pids, top_names = get_top_5_similar(selected_index, flipkart_data_df3, flipkart_id_name)


            # Add a slider in the sidebar to select the Top product
            selected_index = st.sidebar.slider("Select a Top Product:", 1, 5, 1) - 1  # Convert to zero-based index

            # Get the selected product details
            top_pid = top_pids[selected_index]
            top_name = top_names[selected_index]

            # with st.container(border=True):
            st.markdown(f"### {top_name}")

            # Get product details
            top_details = get_product_details(top_pid, flipkart_data_df3)

            # Create a DataFrame for the recommended product details
            top_product_df = pd.DataFrame({
                "Attribute": ["Brand", "Category", "Retail Price", "Discounted Price", "Description"],
                "Details": [
                    str(top_details['brand']),
                    str(top_details['category']),
                    str(top_details['retail_price']),
                    str(top_details['discounted_price']),
                    str(top_details['description'])
                ]
            })

            
            # Ensure all values are strings
            top_product_df["Attribute"] = top_product_df["Attribute"].astype(str)
            top_product_df["Details"] = top_product_df["Details"].astype(str)

            # Display as a table
            st.table(top_product_df)

            # Display Product Images
            top_images = get_product_images(top_pid, flipkart_id_image)
            # st.write(top_images)  # Debugging: Check URLs in cloud

            # Fetch and display images
            img_objects = [fetch_image(img_url) for img_url in top_images]

            img_cols = st.columns(len(img_objects))
            for col, img_obj in zip(img_cols, img_objects):
                if img_obj:
                    col.image(img_obj, width=100)
                else:
                    col.write("Image not available")

            # if top_images:
            #     img_cols = st.columns(len(top_images))  
            #     for col, img_url in zip(img_cols, top_images):
            #         col.image(img_url, width=100)
            # else:
            #     st.write("No images available.")
