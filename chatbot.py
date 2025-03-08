import streamlit as st
import requests
from urllib.parse import urlparse
from mistralai import Mistral

# Mistral Client setup
client = Client(api_key="NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB")

# Dictionary of policies with links
policies = {
    "Academic Annual Leave": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "Academic Appraisal": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
    "Intellectual Property": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    "Credit Hour": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Program Accreditation": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
    "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Graduate Final Grade": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Examination Rules": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "International Student": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "Student Attendance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Graduate Academic Standing": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "Student Engagement": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "Graduate Admissions": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Student Appeals": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Right to Refuse Service": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/right-refuse-service-procedure",
    "Library Study Room Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/library-study-room-booking-procedure",
    "Digital Media Centre Booking": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/digital-media-centre-booking",
    "Use of Library Space": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy"
}

# Function to fetch policy text from URL
def fetch_policy_data(url):
    if not urlparse(url).scheme:
        url = 'https://' + url
    response = requests.get(url)
    response.raise_for_status()  # Raises exception for bad responses
    return response.text

# Function to classify and answer query using Mistral model
def classify_and_answer(query):
    # Step 1: Use the Mistral model to classify the intent of the query
    messages = [{"role": "user", "content": query}]
    response = client.chat.complete(model="mistral-large", messages=messages)
    classified_policy = response['choices'][0]['message']['content']

    # Step 2: Find the corresponding policy URL
    policy_url = policies.get(classified_policy, None)

    if policy_url:
        # Step 3: Fetch and return the policy text
        policy_text = fetch_policy_data(policy_url)
        return f"The policy related to your query is **{classified_policy}**. You can read more about it here: {policy_url}"
    else:
        return "Sorry, I could not find a related policy for your query."

# Streamlit app layout
def streamlit_app():
    st.title("UDST Policy Chatbot")
    
    # Step 1: User Input
    user_query = st.text_input("Enter your query:")
    
    # Step 2: Display answer if user provides input
    if user_query:
        answer = classify_and_answer(user_query)
        st.text_area("Answer", value=answer, height=200)

# Run the app
if __name__ == "__main__":
    streamlit_app()
