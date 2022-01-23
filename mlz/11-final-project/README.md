## Machine Learning Zoomcamp Final Project - R McMaster



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