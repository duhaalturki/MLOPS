import streamlit as st
import mistralai
import os

# Initialize the Mistral client
client = mistralai.Client(api_key='NXyKdE5JFehmTjXn1RtYyVBOlMzPLGyB')  # Replace with your actual API key

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

# Function to answer using Mistral model
def mistral_answer(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context}
    ]
    
    try:
        # Send query to Mistral API
        response = client.chat.complete(model="mistral-large-latest", messages=messages)
        
        # Return the assistant's response
        answer = response['choices'][0]['message']['content']
        return answer
    except mistralai.models.SDKError as e:
        print(f"SDKError: {str(e)}")
        return "Sorry, there was an issue with processing your request."
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return "An unexpected error occurred. Please try again later."

# Function to classify the user's query and map it to the relevant policy
def classify_intent(query):
    query = query.lower()
    for policy_name in policies.keys():
        if policy_name.lower() in query:
            return policy_name, policies[policy_name]
    return None, None  # If no matching policy found

# Streamlit UI
def streamlit_app():
    # Page title
    st.title("UDST Policy Chatbot")

    # Instructions
    st.write("""
        This chatbot can help you with any questions related to UDST policies.
        Please type your question below and hit 'Submit' to get the answer.
    """)

    # Text input for query
    query = st.text_input("Ask a question about UDST policies:")

    # Text area to display the answer
    if st.button("Submit"):
        if query:
            # Classify the intent (which policy is related)
            policy_name, policy_url = classify_intent(query)
            
            if policy_name:
                # If a policy is identified, create the context
                context = f"The policy related to your query is **{policy_name}**. You can read more about it here: {policy_url}"
                
                # Get the answer from the model
                answer = mistral_answer(query, context)
                
                # Display the answer
                st.text_area("Answer", answer, height=200)
            else:
                st.warning("Sorry, I couldn't identify a policy related to your question.")
        else:
            st.warning("Please enter a question to get an answer.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
