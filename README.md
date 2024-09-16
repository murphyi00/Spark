# Spark
Spark distributed processing pipelines SU IST 718 Big Data Analytics
Healthcare Dataset Analysis
Project Overview
This project involves a comprehensive analysis of a healthcare dataset using PySpark in Google Colab. The primary objectives are to:

Perform Exploratory Data Analysis (EDA) and basic visualizations.
Conduct extensive data transformations and feature engineering to prepare the data for machine learning models.
Build and evaluate machine learning models, including Decision Trees and Logistic Regression, to predict health-related outcomes.
Interpret the results and provide insights based on the analysis.
Dataset Description
The dataset used in this project contains various health-related information about individuals, including:

Demographics: State, Sex, Age Category, Race/Ethnicity.
Health Indicators: General Health Status, Physical and Mental Health Days, Sleep Hours, BMI.
Medical History: Presence of diseases like Heart Attack, Angina, Stroke, Asthma, Skin Cancer, COPD, Depression, Kidney Disease, Arthritis, Diabetes.
Lifestyle Factors: Smoking Status, E-Cigarette Usage, Alcohol Consumption, Physical Activities.
Preventive Measures: Last Medical Checkup, Vaccination Status (Flu, Pneumonia, Tetanus), HIV Testing.
COVID-19 Information: COVID-19 Positive Status.
The dataset has been cleaned to remove any missing values (heart_2022_no_nans.csv).

Technologies and Libraries Used
Python: Programming language used for the analysis.
PySpark: For distributed data processing.
Google Colab: Development environment.
# Libraries:
pyspark.sql: For Spark SQL functionalities.
pyspark.ml: Machine learning library in Spark.
pandas: Data manipulation and analysis.
matplotlib, seaborn: Data visualization.
sklearn: For additional machine learning algorithms and evaluation metrics.
# Project Components
1. Data Loading and Exploration
Imported PySpark and initialized SparkSession.
Loaded the healthcare dataset into a Spark DataFrame.
Displayed the schema and performed initial exploration to understand data distribution.
2. Data Transformation and Feature Engineering
Ordinal Encoding: Converted categorical variables with an inherent order into numerical ordinal variables (e.g., Smoker Status, E-Cigarette Usage, Age Category, Last Checkup Time, General Health, Removed Teeth).
Boolean Conversion: Transformed various Yes/No categorical variables into boolean fields for easier modeling (e.g., HadHeartAttack, PhysicalActivities, AlcoholDrinkers).
Binning Numerical Variables: Created buckets for continuous variables like BMI, MentalHealthDays, PhysicalHealthDays, SleepHours, HeightInMeters, and WeightInKilograms using statistical methods (mean and standard deviation).
Handling Skewed Data: Addressed skewness in variables by treating zero and non-zero values separately and creating appropriate bins.
3. Data Visualization
Utilized PySpark DataFrame functions and matplotlib/seaborn for plotting distributions and relationships.
Explored distributions of key variables after transformations to ensure proper binning and encoding.
4. Machine Learning Models
A. Decision Trees
# Model Building:
Used pyspark.ml.classification.DecisionTreeClassifier to predict the target variable (e.g., CovidPos_Boolean).
Selected relevant features based on the analysis.
# Data Preparation:
Assembled features using VectorAssembler.
Split data into training and test sets.
# Model Evaluation:
Evaluated the model using accuracy, precision, recall, and F1-score.
Plotted the feature importance to understand the impact of each variable.
B. Logistic Regression
# Model Building:
Used pyspark.ml.classification.LogisticRegression for classification tasks.
Data Preparation:
Similar to Decision Trees, features were assembled and data was split.
# Model Evaluation:
Assessed the model's performance using ROC curves and AUC metrics.
Interpreted coefficients to understand the relationship between features and the target variable.
5. Results and Interpretation
Identified key factors influencing health outcomes based on model results.
Discussed the implications of findings in a healthcare context.
How to Run This Project
# Prerequisites
Google Colab or a local environment with PySpark installed.
Access to the dataset heart_2022_no_nans.csv.
# Instructions
Clone the Repository or Download the Files:

git clone https://github.com/murphyi00/spark.git
Open the Notebook:

Upload the notebook file (e.g., Healthcare_Dataset_Analysis.ipynb) to Google Colab or open it in Jupyter Notebook.
Install Required Libraries:

In the first cell of the notebook, ensure the installation of PySpark (already included in the code):

python
Copy code
!pip install pyspark
Upload the Dataset:

Upload heart_2022_no_nans.csv to the working directory in Colab or ensure it's accessible in your local environment.
Run the Notebook:

### Execute each cell sequentially to reproduce the analysis.
Project Structure
Healthcare_Dataset_Analysis.ipynb: The main notebook containing all code and explanations.
heart_2022_no_nans.csv: The dataset used for analysis (ensure compliance with data sharing policies before including this in a public repository).
README.md: Project documentation and overview.
### Future Improvements
Feature Expansion: Incorporate additional variables or external datasets to enrich the analysis.
Advanced Modeling: Experiment with other machine learning algorithms like Random Forests, Gradient Boosting, or Neural Networks.
Hyperparameter Tuning: Use techniques like Grid Search or Cross-Validation to optimize model performance.
Deployment: Create a web application or dashboard to make the analysis interactive and accessible to stakeholders.
Data Privacy: Implement measures to ensure data privacy and compliance with regulations like HIPAA if working with sensitive information.
### Conclusions
The analysis provided insights into the health status and risk factors of individuals in the dataset.
Machine learning models identified significant predictors for health outcomes like the likelihood of angina or heart attack.
The project demonstrates the utility of PySpark for handling large datasets and performing complex transformations efficiently.
Contact Information
For questions, suggestions, or collaborations, please reach out via:

Email: murphyi@syr.edu
GitHub: murphyi00
