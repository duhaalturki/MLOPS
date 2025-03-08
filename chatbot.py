import streamlit as st
import mistralai
from mistralai import Client

# Initialize the Mistral client (you need an API key to use this)
client = Client(api_key="NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB")  # Replace with your Mistral API key

# Define the policy dictionary
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

# Function to classify intent and match the policy using Mistral
def classify_intent(query):
    # Use Mistral to classify the intent by querying the model with the policy names
    response = client.chat.complete(
        model="mistral-large-latest",  # Specify the Mistral model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies queries related to policies."},
            {"role": "user", "content": f"Classify the policy related to the query: '{query}'"}
        ]
    )
    intent = response['choices'][0]['message']['content'].strip()
    
    # Return the matching policy and its URL
    for policy_name in policies.keys():
        if policy_name.lower() in intent.lower():
            return policy_name, policies[policy_name]
    return None, None  # If no matching policy is found

# Function to generate answers using the Mistral QA model
def generate_answer(query, policy_url):
    # Create context based on the policy URL
    context = f"Read more about the policy here: {policy_url}. Now answer the following question: {query}"
    response = client.chat.complete(
        model="mistral-large-latest",  # Specify the Mistral model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on policies."},
            {"role": "user", "content": f"Answer the question: '{query}' using the following context: {context}"}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit UI
def streamlit_app():
    st.title("UDST Policy Chatbot")

    # Get query input from the user
    query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if query:
            # Classify the intent and match the query with a policy
            policy_name, policy_url = classify_intent(query)
            if policy_name:
                # Generate answer using Mistral
                answer = generate_answer(query, policy_url)
                st.text_area("Answer", answer, height=200)
            else:
                st.warning("Sorry, no related policy found for your query.")
        else:
            st.warning("Please enter a query.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
