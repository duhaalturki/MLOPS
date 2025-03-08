import streamlit as st
import requests
from transformers import pipeline, MistralForCausalLM, MistralTokenizer

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

# Function to classify intent and map to a policy
def classify_intent(query):
    query = query.lower()
    for policy_name in policies.keys():
        if policy_name.lower() in query:
            return policy_name, policies[policy_name]
    return None, None  # If no matching policy is found

# Function to generate answers using Mistral model
def generate_answer_with_mistral(query, policy_url):
    # Load Mistral model and tokenizer
    model = MistralForCausalLM.from_pretrained("mistral-7b")
    tokenizer = MistralTokenizer.from_pretrained("mistral-7b")

    # Prepare input for the model
    context = f"Context: Please provide an answer based on the policy found at: {policy_url}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(context, return_tensors="pt")
    
    # Generate response
    output = model.generate(inputs['input_ids'], max_length=500)

    # Decode the output to text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
def streamlit_app():
    st.title("UDST Policy Chatbot")

    # Instructions
    st.write("""
        Ask a question related to any UDST policy and get the relevant information.
        Please type your query in the text box below.
    """)

    # Query input box
    query = st.text_input("Enter your query:")

    # Submit button
    if st.button("Submit"):
        if query:
            # Classify the intent to identify which policy is related
            policy_name, policy_url = classify_intent(query)
            
            if policy_name:
                # If the policy is identified, fetch the relevant policy URL and generate an answer
                answer = generate_answer_with_mistral(query, policy_url)
                st.text_area("Answer", answer, height=200)
            else:
                st.warning("Sorry, I couldn't identify a policy related to your question.")
        else:
            st.warning("Please enter a query to get an answer.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
