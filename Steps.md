# Student Performance Indicator


## First install the dependencies

```bash
pip install -r requirements.txt
```
or
```bash
pip install -e .
```

## Steps followed to create this
1. Create **requirements.txt** and in it include `-e .` also so that the package is installed in editable mode when we run pip install -r requirements.txt

2. Create a **setup.py** file with the necessary information about the package, such as name, version, description, author, and the packages to include.

3. Create the **src directory** and add `__init__.py` file to make it a package.
    **Note :**  Remember to include `__init__.py` in all subdirectories that you want to be treated as packages.

4. Created the basic structure including components and pipeline folder.

5. Exception handling and logging is added in the code.
For now automatic loggin implementation in the exception.py file is done

6. EDA and model training is done in the notebook for better understanding and visualization of the data.

7. Next created the data ingestion component which is responsible for reading the data and splitting it into train and test sets.
8. Then created the data transformation component which is responsible for transforming the data and making it ready for model training.
9. Finally created the model training component which is responsible for training the model and saving it

10. Next I created the Predict pipeline which is responsible for loading the model and making predictions on new data.

11. Next main the entry point train_pipeline.py

## To run the train pipeline
```bash
python src/pipeline/train_pipeline.py
```

## To run the Flask app that handles prediction:
```bash
python app.py
```


