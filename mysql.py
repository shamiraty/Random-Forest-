import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
#pip install mysql-connector-python
#pip install mysqlclient



# Connect to the MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="programme_prediction"
)

# Fetch data from the database
cursor = db.cursor(dictionary=True)

# Fetch all programs and their associated subjects
cursor.execute("""
    SELECT p.program, s.subject
    FROM programs p
    JOIN subjects s ON p.id = s.program_id
""")

# Store the data in a pandas DataFrame
data = cursor.fetchall()

# Prepare the dataset similar to before
program_subjects = {}
for row in data:
    program = row["program"]
    subject = row["subject"]
    if program not in program_subjects:
        program_subjects[program] = []
    program_subjects[program].append(subject)

# Prepare DataFrame
df = pd.DataFrame([{"program": program, "subjects": subjects} for program, subjects in program_subjects.items()])
mlb = MultiLabelBinarizer()
subject_matrix = mlb.fit_transform(df['subjects'])
subject_columns = mlb.classes_

# Add subjects as binary features
subject_df = pd.DataFrame(subject_matrix, columns=subject_columns)
df = pd.concat([df, subject_df], axis=1).drop(columns=["subjects"])

# Split Data
X = df[subject_columns]
y = df["program"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI for input
st.title("Program Recommendation System")
st.write("Select subjects you've studied:")

# Create checkboxes for all subjects
subject_checkboxes = {subject: st.checkbox(subject) for subject in subject_columns}

# Get the subjects selected by the user
selected_subjects = [subject for subject, selected in subject_checkboxes.items() if selected]

# If the user has selected subjects, make a prediction
if selected_subjects:
    input_data = [1 if subject in selected_subjects else 0 for subject in subject_columns]
    proba = model.predict_proba([input_data])[0]
    programs_with_proba = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)

    # Display the top 4 recommended programs with probabilities
    st.write("Top 4 Recommended Programs:")
    for program, probability in programs_with_proba[:4]:
        st.write(f"{program}: {probability * 100:.2f}%")
else:
    st.write("Please select at least one subject.")
