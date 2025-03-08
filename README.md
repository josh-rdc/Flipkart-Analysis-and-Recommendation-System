# FLIPKART Analysis and Application

This repository contains the data analysis and possible usecase (Recommendation Application and Discount Prediction) of the Flipkart Dataset.

## Table of Contents
![Asset/Demo_ImageDetection.png](asset/MethodOverview.png)

The whole analysis and development is divided into three main sections, (1) the analysis and preprocessing of dataset, (2) development of the recommendation application, and (3) formulation of the discount prediction model.  

The details taken to perform and formulate the use-cases are presented and discussed in the following sections.
- [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
- [Demo](#demo)
- [Model](#model)
- [Installing Locally](#installing-locally)
- [References](#reference)


## Data Analysis and Preprocessing

### Dataset
Foremost, the dataset is obtained from [kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products/data), a data science platform and online community for data scientists and machine learning practitioners under Google LLC.

As defined in the website, it is a pre-crawled dataset, taken as subset of a bigger [dataset (more than 5.8 million products)](https://www.promptcloud.com/datastock-access-ready-to-use-datasets/?utm_source=fl-kaggle&utm_medium=referral) that was created by extracting data from **Flipkart.com, a leading Indian eCommerce store**.

Features of the dataset presented in the following table:
| Feature                   | Definition                                      |
|---------------------------|------------------------------------------------|
| `product_url`             | The URL of the product on the website.         |
| `product_name`            | The name of the product.                       |
| `product_category_tree`   | The category hierarchy of the product.         |
| `pid`                     | Unique ID assigned by the website.             |
| `retail_price`            | The original price of the product.             |
| `discounted_price`        | The price after applying discounts.            |
| `image`                   | URL(s) of product images.                      |
| `is_FK_Advantage_product` | Indicates if the product has an advantage feature. |
| `description`             | Detailed description of the product.           |
| `product_rating`          | Individual rating given to the product.        |
| `overall_rating`          | Average rating based on user reviews.          |
| `brand`                   | The brand of the product.                      |
| `product_specifications`  | Technical details and specifications.          |

The following features are dropped for this project: 
1. The `crawl_timestamp` as this is the information during the scraping of the dataset and not the actual time it was sold at the website.
2. The `uniq_id` and `product_url` which are website specific information, tagging create by the app and do not necessarily add information to the product.

### Analysis and Preprocessing
1. **Expansion of Category Tree**

    The first process done was to expand the category tree into multiple columns, with main category and limiting up to two (2) sub-categories.
    
2. **Imputation of Null Values**
    
    Checking showed that major features such as retail_price, discounted_price, description, and brand have
    |    pid                  |      0   |
    | ----------------------- | -------- |
    | retail_price            |     78   |
    | discounted_price        |     78   |
    | is_FK_Advantage_product |      0   |
    | description             |      2   |
    | product_rating          |      0   |
    | overall_rating          |      0   |
    | brand                   |   5864   |
    | category_0              |     0    |
    | category_1              |    328   |
    | category_2              |   1457   |


## Demo

The web application was deployed in the Streamlit cloud and can be accessed at [Flipkart Recommendation System](https://flipkart-analysis-and-recommendation-system.streamlit.app/). 

Sample results are shown below:
<details open>
<summary>Sample Images</summary>

#### Detection
![assets/Demo_ImageDetection.png](assets/Demo_ImageDetection.png)

#### Segmentation
![assets/Demo_ImageSegmentation.png](assets/Demo_ImageSegmentation.png)

</details>

<details close>
<summary>Sample Videos</summary>

#### Detection
https://github.com/user-attachments/assets/832283af-ea58-44ef-b8da-a67ed4c0ee55

#### Segmentation
https://github.com/user-attachments/assets/2cedfe91-ea0d-4fab-b610-192d612bc400

</details>

<details close>
   
<summary>Sample Live Video Stream</summary>

#### Segmentation

https://github.com/user-attachments/assets/5d643db0-3a89-4c5f-890b-1a24ab020ade

https://github.com/user-attachments/assets/f9134f57-3ba9-4183-b320-0ec204cbc206

</details>

## Dataset

The dataset used to train the model comprises 24 classes of common grocery items found in the Philippines. These images were manually collected and annotated by the AI 231 class of UP Diliman (AY 2024–2025). To ensure the model's robustness, the dataset includes images captured under various environmental conditions, such as nighttime, daytime, and obstructed views.

<details open>
<summary>Image Dataset Summary</summary>

| #   | Class              | Training Images | Validation Images | Total Images | Specific Brand/Variation                 | Unit of Measurement per Instance            |
|-----|--------------------|-----------------|-------------------|--------------|------------------------------------------|---------------------------------------------|
| 1   | Bottled Soda       | 477             | 58               | 535          | Coca-Cola (Coke Zero)                    | 1.9L bottle, 320mL bottle                   |
| 2   | Cheese             | 310             | 40               | 350          | Eden (Classic)                           | 165g box, 45g pack                          |
| 3   | Chocolate          | 459             | 59               | 518          | KitKat (Chocolate)                       | 36.5g pack                                  |
| 4   | Coffee             | 404             | 41               | 445          | Nescafe Original (Classic)               | 52g twin pack, 28g pack                     |
| 5   | Condensed Milk     | 370             | 46               | 416          | Alaska (Classic)                         | 208g can                                    |
| 6   | Cooking Oil        | 467             | 55               | 522          | Simply Canola Oil                        | 1L bottle                                   |
| 7   | Corned Beef        | 442             | 58               | 500          | Purefoods (Classic, Spicy)               | 150g, 210g, 380g can                        |
| 8   | Garlic             | 317             | 33               | 350          | Whole                                    | Whole                                       |
| 9   | Instant Noodles    | 431             | 42               | 473          | Lucky Me! (Sweet and Spicy)              | 80g pack                                    |
| 10  | Ketchup            | 477             | 47               | 524          | UFC (Banana)                             | 530g                                        |
| 11  | Lemon              | 324             | 38               | 362          | Whole                                    | Whole                                       |
| 12  | All-purpose Cream  | 451             | 49               | 500          | Nestle (Classic)                         | 250g box                                    |
| 13  | Mayonnaise         | 319             | 31               | 350          | Lady's Choice (Classic)                  | 700mL bottle                                |
| 14  | Peanut Butter      | 485             | 35               | 520          | Lady's Choice, Skippy                    | 170g, 340g bottle                           |
| 15  | Pasta              | 443             | 57               | 500          | Royal Linguine                           | 1kg pack                                    |
| 16  | Pineapple Juice    | 449             | 50               | 499          | Del Monte (Fiber, ACE)                   | 240mL can                                   |
| 17  | Crackers           | 462             | 47               | 509          | Skyflakes, Rebisco                       | 22g, 33g pack                               |
| 18  | Sardines (Canned)  | 305             | 45               | 350          | 555 (Tomato)                             | 155g can                                    |
| 19  | Pink Shampoo       | 444             | 56               | 500          | Sunsilk (Smooth and Manageable)          | 180mL bottle                                |
| 20  | Soap               | 446             | 54               | 500          | Dove (Lavender)                          | 106g box                                    |
| 21  | Soy Sauce          | 452             | 48               | 500          | Silverswan                                | 385mL bottle                                |
| 22  | Toothpaste         | 456             | 44               | 500          | Colgate (Advanced White)                 | 160g box                                    |
| 23  | Canned Tuna        | 461             | 61               | 522          | Century Tuna (Original, Hot and Spicy)   | 155g, 180g can                              |
| 24  | Alcohol            | 426             | 34               | 460          | Green Cross (Ethyl)                      | 500ml bottle                                |

</details>

For dataset access requests, please feel free to contact me.

## Model  

The application utilizes **YOLO11** by Ultralytics, the latest iteration in the YOLO series of real-time object detectors. YOLOv11 redefines object detection with cutting-edge accuracy, speed, and efficiency.  

### Model Variations  
YOLOv11 offers multiple variations to suit a wide range of tasks:  
- **Nano (n)**: Optimized for lightweight deployment.  
- **Small (s)**: Balances speed and accuracy for real-time use.  
- **Medium (m)**: Enhanced performance for more complex scenarios.  
- **Large (l)**: High-accuracy models for demanding tasks.  
- **Xtra-large (x)**: Maximum accuracy for advanced applications.  

Supported tasks include:  
- Object detection  
- Segmentation  
- Classification  
- Pose estimation  
- Oriented object detection  

For more details on model parameters and performance benchmarks (e.g., on the COCO dataset), visit the [Ultralytics YOLO11 Model Card](https://docs.ultralytics.com/models/yolo11).  

### Model Training
Models are trained on NVIDIA A100-SXM4-40GB GPU. Training time ranges from ~1 to 10 hours depending on the model variation and configuration.

## Installing Locally

To run this project locally, please follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/josh-rdc/grocery-detection-segmentation-webapp
   ```

2. Navigate to the project folder:

   ```
   cd grocery-detection-segmentation-webapp
   ```

3. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:

   ```
   streamlit run 🏠_Home.py
   ```

## Reference

```
@software{Jocher_Ultralytics_YOLO_2023,
author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
license = {AGPL-3.0},
month = jan,
title = {{Ultralytics YOLO}},
url = {https://github.com/ultralytics/ultralytics},
version = {8.0.0},
year = {2023}
}
```

## Contact

If you find this work useful, kindly give a star ⭐ this repository. 

For any inquiries, feel free to contact me through the following:
 <p>
<a href="mailto:delacruz.joshua.reyes@gmail.com"><img src="https://img.shields.io/badge/Email-c14438?&logo=gmail&logoColor=white" alt="EmailId" height="20"/></a>
<a href="https://www.linkedin.com/in/joshreyesdelacruz/" target="blank"><img src="https://img.shields.io/badge/Linkedin-%230077B5.svg?logo=linkedin&logoColor=white" alt="LinkedinId" height="20" /></a>

</p>
