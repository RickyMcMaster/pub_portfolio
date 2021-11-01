## Machine Learning Zoomcamp Midterm Project - R McMaster

I struggled for some time to find a dataset that looked useable and suitable to my current knowledge level.

After browsing through datasets in various places, but particularly Kaggle, I finally came upon [this one](https://www.kaggle.com/HRAnalyticRepository/employee-attrition-data), which focuses on employee attrition.

I decided upon this because first of all I found it relatable in light of what we had previously studied for customer churn, secondly because my first data job was in HR Information so it was personally interesting, and thirdly because it contains some interesting features (e.g. multiple rows per employee) and possibilities for feature engineering.

The data in the dataset relates to a ten-year period in the history of a fictional company, with one row per year for each employee (though of course, some have an employment history a lot shorter than 10 years).

The goal of the project will be therefore to train a model on this data, and use it to predict whether or not an employee is likely to churn when new customer information is provided.  This information can be provided via a json file - I have included a test json file for this purpose.

Files included are as follows:

* Main workbook: MLZ-Assignment07.ipynb
* train.py, which trains and exports the model to resig-model.bin
* predict.py, which should take an individual employee and produce a prediction
* Pipfile which includes necessary libraries for the model
* Dockerfile - uses the aforementioned pipfile and exposes port 9696 for the webservice (using Flask)
    * Build & Run this with: 
        docker build -t resignation-model .
        docker run -it -p 9696:9696 --rm resignation-model 