import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
from rapidfuzz import process  # Using RapidFuzz for fuzzy matching

# Connect to the MySQL database
db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="programme_prediction"
)

# Fetch data from the database
cursor = db.cursor()

# Fetch all programs and their associated subjects
cursor.execute("""
    SELECT p.name AS program, s.name AS subject
    FROM Programs p
    JOIN Subjects s ON p.id = s.program_id;
""")

# Store the results in a DataFrame
program_subject_data = cursor.fetchall()
df = pd.DataFrame(program_subject_data, columns=["program", "subject"])

# Prepare the data for the model
mlb = MultiLabelBinarizer()
subject_matrix = mlb.fit_transform(df['subject'])
subject_columns = mlb.classes_

# Add subjects as binary features
subject_df = pd.DataFrame(subject_matrix, columns=subject_columns)
df = pd.concat([df, subject_df], axis=1)

# Split Data
X = df[subject_columns]
y = df["program"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Random Forest Classifier")
st.subheader("The Classification and Regression Trees (CART) Algorithm")  # Subtitle
st.write("Enter the subjects you have completed in high school, and we will recommend the top undergraduate programs for you.")


# Calculate the accuracy of the model on the test data
accuracy = model.score(X_test, y_test)

# Display the accuracy in the Streamlit app
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")


with st.form(key="subject_form"):
    subjects_input = st.text_input("Enter subjects (comma separated)", "")
    submit_button = st.form_submit_button(label="Get Recommendations")

def is_valid_subjects(input_subjects):
    # Check if all input subjects are valid fuzzy matches
    valid_subjects = df['subject'].tolist()
    for input_subject in input_subjects:
        best_match = process.extractOne(input_subject, valid_subjects)
        if best_match[1] < 80:  # Match threshold (can be adjusted)
            return False
    return True

if submit_button and subjects_input:
    student_subjects = [subject.strip() for subject in subjects_input.split(",")]

    # Ensure that all subjects are unique
    student_subjects = list(set(student_subjects))  # Remove duplicates

    # Ensure that the user enters at least 2 subjects
    if len(student_subjects) < 2:
        st.error("Please enter at least 2 subjects.")
    else:
        # Validate that all input subjects are valid fuzzy matches
        if not is_valid_subjects(student_subjects):
            st.error("Some input subjects are invalid or don't closely match any existing subjects.")
        else:
            # Convert the valid input to binary format
            input_data = [1 if subject in student_subjects else 0 for subject in subject_columns]
            proba = model.predict_proba([input_data])[0]
            programs_with_proba = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)

            st.write("Top 6 Recommended Programs:")
            for program, probability in programs_with_proba[:12]:
             st.warning(f"{program}:  [   {probability * 100:.1f}%  ]")

