# Student Performance Analyser

A full Data Science pipeline project that loads a student dataset, cleans it, performs exploratory data analysis (EDA), trains a classification model to predict pass/fail, and deploys a basic interactive dashboard with Streamlit.

## Setup Instructions

1.  **Clone or download the repository.**
2.  **Install dependencies:**
    ```bash
    python -m venv venv
    .\\venv\\Scripts\\activate
    pip install -r requirements.txt
    ```
3.  **Data Acquisition:**
    Download a student performance dataset (e.g., from Kaggle) and place the CSV file in the `data/` directory named as `student_data.csv`.
4.  **Run Pipeline (Cleaning, EDA, Modeling):**
    Execute the data science pipeline to clean data and train the model.
    ```bash
    python execute_pipeline.py
    ```
    *Note: The Jupyter notebooks in the `notebooks/` directory can also be explored interactively.*
5.  **Run the Streamlit Dashboard:**
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Project Structure
- `notebooks/`: Contains Jupyter notebooks with code for data cleaning, EDA, and model training.
- `execute_pipeline.py`: The main script to run the data pipeline and generate the model.
- `data/`: Stores raw (`student_data.csv`) and cleaned (`cleaned_student_data.csv`) datasets.
- `app/`: Contains the Streamlit application code and exported ML model artifacts.

## Screenshots
*(Add screenshots of the Streamlit dashboard here after deployment)*
