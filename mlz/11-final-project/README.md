## Machine Learning Zoomcamp Final Project - R McMaster

This project is based on a house price dataset I found on Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/description

It has 79 features so this seemed promising in terms of data richness.

Please see the data_description.txt which I copied for reference - however, I have developed the file a bit further as you will see in the notebook.

Files included are as follows:

* Main workbook: notebook.ipynb
* train.py, which trains and exports the model to houses-model.bin
* predict.py, which should take an individual house and produce a prediction
* Pipfile which includes necessary libraries for the model
* Dockerfile - uses the aforementioned pipfile and exposes port 9696 for the webservice (using Flask)
    * Build & Run this with: 
        ```bash
        docker build -t resignation-model .
        ```
        ```bash
        docker run -it -p 9696:9696 --rm houses-model
        ```
