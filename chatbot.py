import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MistralForConditionalGeneration, MistralTokenizer

# Define the UDST policies
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

# Load the Mistral model and tokenizer
tokenizer = MistralTokenizer.from_pretrained("facebook/mistral-7b-v0.1")
model = MistralForConditionalGeneration.from_pretrained("facebook/mistral-7b-v0.1")

# Vectorizer for intent classification
vectorizer = TfidfVectorizer()
policy_names = list(policies.keys())
X = vectorizer.fit_transform(policy_names)

# Function to classify the intent (which policy the query relates to)
def classify_intent(query):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, X)
    best_match_index = similarity_scores.argmax()
    return policy_names[best_match_index]

# Function to generate answers using Mistral (RAG-based QA)
def get_policy_answer(query, policy_name):
    # Preparing the context based on the policy
    context = f"The policy for {policy_name} is available at {policies[policy_name]}."
    
    # Encode the inputs for Mistral
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    context_inputs = tokenizer(context, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Generate the answer using Mistral
    outputs = model.generate(input_ids=inputs['input_ids'], context_input_ids=context_inputs['input_ids'])
    
    # Decode the generated output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app interface
def app():
    st.title("UDST Policy Chatbot")
    st.write("Ask me any questions about UDST policies!")

    # User query input
    query = st.text_input("Enter your question:")

    answer = ""
    if query:
        # Classify the intent (which policy the query relates to)
        policy_name = classify_intent(query)

        # Get the answer using Mistral-based QA pipeline
        answer = get_policy_answer(query, policy_name)

        # Display the answer in the text area
        st.text_area("Answer:", answer, height=200)

# Run the app
if __name__ == "__main__":
    app()

