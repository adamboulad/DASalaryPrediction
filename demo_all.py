import numpy as np
import streamlit as st

import pickle

# Load the trained models and encoders using pickle
try:
    with open('salary_prediction_avg_model.pkl', 'rb') as f:
        model_avg = pickle.load(f)

    with open('salary_prediction_min_model.pkl', 'rb') as f:
        model_min = pickle.load(f)

    with open('salary_prediction_max_model.pkl', 'rb') as f:
        model_max = pickle.load(f)

    with open('level_encoder.pkl', 'rb') as f:
        le_level = pickle.load(f)

    with open('role_encoder.pkl', 'rb') as f:
        le_role = pickle.load(f)

    print("All models and encoders have been successfully loaded.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all pickle files are in the current directory.")
except Exception as e:
    print(f"An error occurred while loading the files: {e}")

# Ordered levels
ordered_levels = ['Associate', 'Junior', 'Mid', 'Senior', 'Lead', 'Manager']

# Predefined list of roles (can be expanded based on your use case)
roles_list = ['.NET Developer', 'Accounting', 'Android Developer', 'Business Analyst', 'Call Center', 'Data Engineer', 'Data Scientist', 'DevOps Engineer', 'Flutter Developer', 'Frontend Developer', 'HR', 'iOS Developer', 'Java Developer', 'Machine Learning Engineer', 'PHP Developer', 'Presales Officer', 'Product Manager', 'Product Owner', 'Project Manager', 'Python Developer', 'Quality Assurance Engineer', 'R&D', 'Regional Manager', 'RPA Developer', 'Sales Account Manager', 'SysAdmin Engineer', 'System Analyst', 'Technical Support Engineer', 'UI/UX Engineer']


# Define experience ranges for each level
experience_ranges = {
    'Associate': (0, 2),
    'Junior': (0, 4),
    'Mid': (2, 7),
    'Senior': (4, 10),
    'Lead': (7, 15),
    'Manager': (8, 20),
}

# Function to handle unseen roles/levels
def handle_unseen_category(encoder, category, category_type):
    if category not in encoder.classes_:
        st.warning(f"Unseen {category_type} '{category}' detected. Using a default category for prediction.")
        return encoder.transform([encoder.classes_[0]])[0]
    return encoder.transform([category])[0]

# Function to predict salaries
def predict_salaries(years_of_experience, level, role):
    # Handle unseen levels and roles
    level_encoded = handle_unseen_category(le_level, level, "level")
    role_encoded = handle_unseen_category(le_role, role, "role")
    
    # Prepare the feature array for prediction
    features = np.array([[years_of_experience, level_encoded, role_encoded]])
    
    # Predict salaries using the trained models
    predicted_avg_salary = model_avg.predict(features)[0]
    predicted_min_salary = model_min.predict(features)[0]
    predicted_max_salary = model_max.predict(features)[0]
    
    return predicted_avg_salary, predicted_min_salary, predicted_max_salary

# Streamlit UI
st.title("Salary Prediction App")
st.write("This app predicts min, average, and max salary based on years of experience, level, and role.")

# Select level
level = st.selectbox("Select Level", ordered_levels)

# Show the valid range of experience for the selected level
min_years, max_years = experience_ranges.get(level, (0, 20))
st.write(f"For {level} level, you can choose between {min_years} and {max_years} years of experience.")

# Input years of experience with dynamic range
years_of_experience = st.slider("Years of Experience", min_years, max_years, min_years)

# Role input as a dropdown
role = st.selectbox("Select Role", roles_list)

# Button to make prediction
if st.button("Predict Salary"):
    try:
        # Make the prediction
        predicted_avg_salary, predicted_min_salary, predicted_max_salary = predict_salaries(years_of_experience, level, role)
        st.success(f"Predicted Salaries for {years_of_experience} years of experience, {level}, {role}:")
        st.write(f"**Minimum Salary**: ${predicted_min_salary:.2f}")
        st.write(f"**Average Salary**: ${predicted_avg_salary:.2f}")
        st.write(f"**Maximum Salary**: ${predicted_max_salary:.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
