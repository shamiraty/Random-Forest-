## RANDOM FOREST CLASSIFIER ML
### THE CLASSIFICATION AND REGRESSION TREES (CART) ALGORITHM
#### HIGH SCHOOL SUBJECT TO PROGRAM RECOMMENDATION

<img src="https://github.com/user-attachments/assets/90e6bb44-455c-4079-9a4d-59feadce6fdb" alt="logo7" width="100" height="100" />

- **Location**: Dodoma, Tanzania
- **Email**: [sashashamsia@gmail.com](mailto:sashashamsia@gmail.com)
- **WhatsApp**: [+255675839840](https://wa.me/255675839840)
- **Demo**: [Online](https://random-forest-tree.streamlit.app/)
  <!--- **Youtube**: [Videos](https://www.youtube.com/channel/UCjepDdFYKzVHFiOhsiVVffQ)-->

| Icon | Rank | Professional Target Audience                                |
|------|------|------------------------------------------------------------|
| 🔬  | 1    | Healthcare Researchers (Epidemiologists, Clinical Researchers) |
| 🩺  | 2    | Physicians (General Practitioners and Specialists)            |
| 🏛️  | 3    | Public Health Officials (Health Department Personnel, Policy Makers) |
| 🏢  | 4    | Hospital Administrators                                      |
| 👩‍⚕️ | 5    | Nurses                                                     |
| 📊  | 6    | Data Analysts in Healthcare                                   |
| 👨‍💼 | 7    | Medical Directors                                          |
| 💊  | 8    | Pharmacists                                                 |
| 🏥  | 9    | Clinic Administrators                                        |
| 🎓  | 10   | Medical Educators (Professors of medicine)                   |

## Disclaimer

> The patient records used to train and test the **Symptom-Matcher AI** model are **not real patient data**. They are **fictitious data** that have been generated for educational purposes only. These records have no connection to any actual individuals or real-life medical conditions. 

> The diseases and symptoms displayed in the application are **for learning and demonstration purposes**. They do not represent actual medical diagnoses and should not be interpreted as such. The use of these simulated cases is intended solely for academic and training purposes.

> This application and its content are not intended to diminish or disrespect the real-world medical field, institutions, or individuals. The information presented is purely hypothetical and should not be used for making medical decisions. The primary goal of this project is to provide a platform for learning and development in the field of machine learning and healthcare technology.

## Introduction

The **Random Forest Classifier** is a robust and versatile machine learning algorithm that uses an ensemble of decision trees to make predictions. In this application, the model is used to recommend undergraduate programs based on the subjects completed by students in high school. The system leverages data from a **CSV file**, employs **fuzzy string matching** to handle variations in subject names, and provides users with a personalized recommendation of programs they might be suited for.

---

## Problem Statement

Students often face challenges when choosing undergraduate programs, as they may not know which programs align with their completed high school subjects. This decision-making process is further complicated by variations in how subjects are named or entered, such as typographical errors or inconsistent naming conventions. This app aims to solve this problem by providing program recommendations based on high school subjects and correcting any discrepancies in the user input.

---

## Main Objective

The main objective of this project is to develop an application that:
1. Recommends top undergraduate programs based on the high school subjects entered by the student.
2. Provides accurate results even if the user makes slight errors or variations in subject names, thanks to fuzzy matching.
3. Uses the **Random Forest Classifier** to generate recommendations based on subject patterns in the dataset.

---

## Specific Objectives

1. To preprocess and prepare data from a CSV file into a format suitable for machine learning.
2. To train a **Random Forest Classifier** model that can predict the best programs for a student based on their subjects.
3. To validate the entered subjects using fuzzy string matching to ensure that subject names are correctly recognized.
4. To display the top 5 recommended programs along with their prediction probabilities.

---

## Methodology

The application follows these steps:

1. **Data Collection and Preprocessing**:
   - Data is loaded from a CSV file containing information about `program` and `subject` columns.
   - The subject data is normalized to lowercase to maintain uniformity.
   - The `program` and `subject` data are then transformed into a one-hot encoded format where each subject is represented as a binary vector.

2. **Training the Random Forest Model**:
   - The data is split into training and testing sets. A **Random Forest Classifier** is trained using the subjects as features and the programs as the target.
   - The model is evaluated on the test set, ensuring it can make accurate predictions.

3. **Fuzzy Matching for Subject Validation**:
   - When a user enters their high school subjects, the input is matched against the available subjects in the CSV data using the **fuzzywuzzy** library. This ensures that minor errors in subject names do not affect the prediction.

4. **Program Recommendation**:
   - Once the subjects are validated, the model predicts the likelihood of the student being successful in different programs.
   - The top 5 recommended programs are displayed based on the highest predicted probabilities.

---

## Methodology Details

### 1. **Data Loading and Preprocessing**
   The application loads data from a CSV file containing both `program` and `subject` data. The data is preprocessed by converting all subjects to lowercase to avoid discrepancies due to case sensitivity. The subjects are then encoded using a **MultiLabelBinarizer** to represent each subject as a binary vector.

### 2. **Data Preparation**
   - A **MultiLabelBinarizer** is used to convert the subjects into binary features, suitable for training the Random Forest model.
   - The data is split into training and testing sets, and the model is trained to predict which program a student is most likely suited for, given their subjects.

### 3. **Model Training**
   The **Random Forest Classifier** is trained on the prepared data. It learns to associate a set of subjects with a particular program. The model is trained using scikit-learn's `train_test_split` for data splitting and `RandomForestClassifier` for classification.

### 4. **Fuzzy Matching**
   To handle subject variations or typos, **fuzzywuzzy** is employed to match the user’s input with the nearest valid subjects in the CSV data. If the match score is below a certain threshold (e.g., 70%), an error message is displayed.

### 5. **User Interaction via Streamlit**
   - **User Input**: The user enters their completed subjects separated by commas.
   - **Recommendations**: Upon submitting the subjects, the model outputs the top 5 recommended programs along with their prediction probabilities.

---

## Technologies Used

- **Streamlit**: Used for creating an interactive and easy-to-use frontend for the application.
- **Pandas**: For data manipulation and handling the CSV file.
- **scikit-learn**: For implementing the Random Forest Classifier and preprocessing the data.
- **fuzzywuzzy**: Used for fuzzy string matching to account for variations or typos in subject names.

---

## Expected Outcome

The application provides students with an easy way to enter their high school subjects and receive program recommendations. The top 5 recommended programs will be shown with their respective probabilities, allowing students to make an informed decision about their next academic steps.

---

## Limitations

- The system relies heavily on the data available in the CSV file. If the data is sparse or not representative, the model's recommendations may not be accurate.
- The fuzzy matching technique might not be able to match every possible variation of a subject name, depending on how distinct the subject names are.

---

## Future Plans

- Integrating more subject-related features such as subject difficulty or interest areas could improve the program recommendations.
- The addition of more robust error handling and validation features would ensure a better user experience.
- Potentially expand the model to handle other factors like student preferences or geographical constraints for a more personalized recommendation system.

---

## Conclusion

The **Random Forest Classifier** model combined with **fuzzy string matching** provides an effective solution to recommending programs based on high school subjects. By leveraging machine learning and fuzzy matching, the application offers accurate and personalized program recommendations, helping students make better-informed decisions about their academic future.
