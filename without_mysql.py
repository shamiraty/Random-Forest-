import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Dataset definition
data = [
    {"program": "Business Administration", "subjects": ["History", "Geography", "Maths", "Civics", "Bookkeeping", "Commerce", "Economics", "Management", "history","english","Leadership"]},
    {"program": "Information Technology", "subjects": ["Maths", "Physics", "Biology", "Electronics", "Geography", "Programming", "Web Development", "Networking","mathematics"]},
    {"program": "Medicine", "subjects": ["Biology", "Chemistry", "Physics", "Maths", "General Studies", "Anatomy", "Physiology", "Biostatistics", "Pharmacology"]},
    {"program": "Engineering", "subjects": ["Maths", "Physics", "Chemistry", "Technical Drawing", "Electronics", "Mechanics", "Material Science", "Design"]},
    {"program": "Law", "subjects": ["History", "Civics", "English Literature", "General Studies", "Geography", "Constitutional Law", "Criminal Law", "Ethics"]},
    {"program": "Computer Science", "subjects": ["Maths", "Physics", "Programming", "Electronics", "Geography", "Data Structures", "AI", "Machine Learning", "Algorithms"]},
    {"program": "Economics", "subjects": ["Maths", "Economics", "Civics", "History", "Geography", "Statistics", "Microeconomics", "Macroeconomics"]},
    {"program": "Agriculture", "subjects": ["Biology", "Chemistry", "Agriculture", "Geography", "Physics", "Soil Science", "Horticulture", "Crop Production"]},
    {"program": "Environmental Science", "subjects": ["Biology", "Geography", "Environmental Studies", "Chemistry", "Physics", "Ecology", "Climate Change", "Sustainability"]},
    {"program": "Biotechnology", "subjects": ["Biology", "Chemistry", "Genetics", "Microbiology", "Molecular Biology", "Bioinformatics", "Physics", "Maths"]},
    {"program": "Cyber Security", "subjects": ["Programming", "Networking", "Ethical Hacking", "Maths", "Physics", "Cryptography", "Information Security", "AI"]},
    {"program": "Digital Forensics", "subjects": ["Networking", "Law", "Maths", "Programming", "Investigation Techniques", "Cyber Crime", "Data Recovery", "Digital Systems"]},
    {"program": "Education", "subjects": ["Psychology", "Sociology", "Teaching Methods", "General Studies", "English", "Geography", "Curriculum Development", "Research Methods"]},
    {"program": "Tourism and Hospitality", "subjects": ["Geography", "Civics", "History", "Management", "Marketing", "Event Management", "Cultural Studies", "Economics"]},
    {"program": "Renewable Energy", "subjects": ["Physics", "Maths", "Environmental Studies", "Chemistry", "Renewable Resources", "Energy Systems", "Design", "Economics"]},
    {"program": "Multimedia Technology", "subjects": ["Graphics Design", "Video Editing", "Programming", "Physics", "Maths", "Web Design", "Animation", "Human-Computer Interaction"]},
    {"program": "Art in Economics and Statistics", "subjects": ["Economics", "Statistics", "Mathematics", "Civics", "Geography"]},
    {"program": "Art in Environmental Economics and Policy", "subjects": ["Environmental Studies", "Economics", "Policy Making", "Geography", "Civics"]},
    {"program": "Arts in Economics", "subjects": ["Economics", "Mathematics", "History", "Geography", "Philosophy"]},
    {"program": "Arts in Economics and Sociology", "subjects": ["Economics", "Sociology", "Psychology", "Mathematics", "History"]},
    {"program": "Business Administration", "subjects": ["Accounting", "Marketing", "Finance", "Human Resources", "Entrepreneurship"]},
    {"program": "Commerce in Accounting", "subjects": ["Accounting", "Auditing", "Taxation", "Corporate Law", "Finance"]},
    {"program": "Commerce in Entrepreneurship", "subjects": ["Entrepreneurship", "Marketing", "Business Strategy", "Finance", "Accounting"]},
    {"program": "Commerce in Finance", "subjects": ["Finance", "Economics", "Investments", "Banking", "Taxation"]},
    {"program": "Commerce in Human Resource Management", "subjects": ["Human Resources", "Organizational Behavior", "Labor Laws", "Psychology", "Communication Skills"]},
    {"program": "Commerce in International Business", "subjects": ["International Trade", "Global Economics", "Foreign Policy", "Marketing", "Supply Chain"]},
    {"program": "Commerce in Marketing", "subjects": ["Marketing", "Digital Marketing", "Brand Management", "Consumer Behavior", "Sales"]},
    {"program": "Education in Administration and Management", "subjects": ["Educational Management", "Leadership", "Policy Planning", "Human Resources", "Statistics"]},
    {"program": "Education in Adult Education and Community", "subjects": ["Adult Learning", "Community Development", "Policy Implementation", "Sociology", "Psychology"]},
    {"program": "Education in Arts", "subjects": ["Fine Arts", "History", "Literature", "Philosophy", "Psychology"]},
    {"program": "Education in Science", "subjects": ["Biology", "Chemistry", "Physics", "Mathematics", "Environmental Studies"]},
    {"program": "Science in Computer Engineering", "subjects": ["Programming", "Electronics", "Networking", "System Design", "Artificial Intelligence"]},
    {"program": "Science in Cyber Security", "subjects": ["Cyber Security", "Digital Forensics", "Ethical Hacking", "Data Privacy", "Information Security"]},
    {"program": "Science in Renewable Energy Engineering", "subjects": ["Renewable Energy", "Environmental Science", "Physics", "Chemistry", "Engineering Mathematics"]},
    {"program": "Science in Applied Geology", "subjects": ["Geology", "Geophysics", "Environmental Science", "Mining", "Geochemistry"]},
    {"program": "Science in Statistics", "subjects": ["Probability", "Statistical Analysis", "Mathematics", "Data Science", "Economics"]},
    {"program": "Science in Software Engineering", "subjects": ["Software Development", "Algorithms", "Database Systems", "Operating Systems", "Cloud Computing"]},
    {"program": "Science in Telecommunications Engineering", "subjects": ["Telecommunications", "Signal Processing", "Networking", "Electronics", "Wireless Systems"]},
    {"program": "Philosophy in Economics", "subjects": ["Advanced Economics", "Economic Theory", "Research Methodology", "Quantitative Analysis", "Policy Development"]},
    {"program": "Philosophy in Business Administration", "subjects": ["Business Strategy", "Leadership", "Corporate Governance", "Financial Management", "Operations Management"]},
]


# Prepare DataFrame
df = pd.DataFrame(data)
mlb = MultiLabelBinarizer()
subject_matrix = mlb.fit_transform(df['subjects'])
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
# Streamlit App
st.title("Random Forest Classifier")
st.subheader("The Classification and Regression Trees (CART) Algorithm")  # Subtitle
st.write("Enter the subjects you have completed in high school, and we will recommend the top undergraduate programs for you.")



with st.form(key="subject_form"):
    subjects_input = st.text_input("Enter subjects (comma separated)", "")
    submit_button = st.form_submit_button(label="Get Recommendations")

def validate_subjects(input_subjects):
    """Check if all entered subjects are valid."""
    valid_subjects = set(subject_columns)
    return all(subject in valid_subjects for subject in input_subjects)

if submit_button and subjects_input:
    student_subjects = [subject.strip() for subject in subjects_input.split(",")]

    # Ensure that at least 2 subjects are entered
    if len(student_subjects) < 2:
        st.error("Please enter at least 2 subjects.")
    else:
        # Validate entered subjects
        if not validate_subjects(student_subjects):
            st.error("Some entered subjects are not recognized. Please check your inputs.")
        else:
            # Prepare input data for prediction
            input_data = [1 if subject in student_subjects else 0 for subject in subject_columns]
            proba = model.predict_proba([input_data])[0]
            programs_with_proba = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)

            st.write("Top 5 Recommended Programs:")
            for program, probability in programs_with_proba[:5]:
             st.warning(f"{program}:  [   {probability * 100:.1f}%  ]")
