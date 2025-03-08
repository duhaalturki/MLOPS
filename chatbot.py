import streamlit as st

# Define a simple policy dictionary
policies = {
    "Academic Annual Leave": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "Academic Appraisal": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy"
}

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
                # Display the answer
                st.text_area("Answer", context, height=200)
            else:
                st.warning("Sorry, I couldn't identify a policy related to your question.")
        else:
            st.warning("Please enter a question to get an answer.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
