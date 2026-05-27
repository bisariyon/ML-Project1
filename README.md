---
title: Ml Students Performance
emoji: 💻
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Student Performance Indicator

The Student Performance Indicator project predicts student mathematics performance based on various features such as:

- Gender
- Race/Ethnicity
- Parental level of education
- Lunch type
- Test preparation course
- Reading score
- Writing score

This project demonstrates a complete Machine Learning lifecycle including:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation
- Prediction Pipeline
- Flask Application Deployment
- Docker Containerization
- Hugging Face Deployment

The goal of this project is to understand how different factors affect student academic performance and build a predictive system using Machine Learning techniques.

## Live Application

🚀 [Open Live App](https://bisariyon-ml-students-performance.hf.space/predictdata)

---

## How to Run This Project Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
pip install -e .
```

---

## Steps Followed to Build This Project

1. Created a **requirements.txt** file and included `-e .` so the package installs in editable mode when running:

```bash
pip install -r requirements.txt
```

2. Created a **setup.py** file with package information such as:
   - name
   - version
   - description
   - author
   - packages to include

3. Created the **src** directory and added `__init__.py` files to make Python treat directories as packages.

> **Note:** Add `__init__.py` in all subdirectories that should behave as packages.

4. Created the project structure including:
   - components
   - pipeline
   - logger
   - exception handling

5. Added logging and custom exception handling for better debugging and tracking.

6. Performed EDA and model training inside Jupyter notebooks for better understanding and visualization.

7. Created the **Data Ingestion Component** responsible for:
   - reading the dataset
   - splitting train/test data
   - saving artifacts

8. Created the **Data Transformation Component** responsible for:
   - preprocessing data
   - feature engineering
   - creating transformation pipelines

9. Created the **Model Trainer Component** responsible for:
   - training models
   - evaluating models
   - saving the best model

10. Created the **Prediction Pipeline** responsible for:

- loading the trained model
- making predictions on new data

11. Created the training entry point:

```bash
src/pipeline/train_pipeline.py
```

---

## Run the Training Pipeline

```bash
python src/pipeline/train_pipeline.py
```

---

## Run the Flask Application

```bash
python app.py
```
