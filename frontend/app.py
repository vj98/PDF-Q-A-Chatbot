import streamlit as st
import requests
import time  # To measure response time

# FastAPI server URL
API_URL = "http://backend:10000"  # Make sure this matches the FastAPI server

# Streamlit UI components
st.title("Document QA System with Metrics")

# Metrics placeholders
response_time_placeholder = st.empty()  # Placeholder to display response time
accuracy_placeholder = st.empty()       # Placeholder to display accuracy rating
satisfaction_placeholder = st.empty()   # Placeholder to display user satisfaction

# File upload section
st.header("Upload PDF File")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Add loader while file is being uploaded
    with st.spinner("Uploading file..."):
        # Upload file to the FastAPI backend
        files = {"uploaded_file": (uploaded_file.name, uploaded_file, "application/pdf")}
        start_time = time.time()  # Start timing the request
        response = requests.post(f"{API_URL}/upload/file", files=files)
        end_time = time.time()  # End timing

        # Measure response time
        response_time = end_time - start_time
        response_time_placeholder.write(f"Upload Response Time: {response_time:.2f} seconds")

    if response.status_code == 200:
        st.success("File successfully uploaded and ingested.")
    else:
        st.error(f"Failed to upload file. Status code: {response.status_code}")

# Query section
st.header("Query the Document")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        with st.spinner("Retrieving the answer..."):
            # Send the query to FastAPI backend
            start_time = time.time()  # Start timing the query
            params = {"query": query}
            response = requests.get(f"{API_URL}/query", params=params)
            end_time = time.time()  # End timing

            # Measure response time
            response_time = end_time - start_time
            response_time_placeholder.write(f"Query Response Time: {response_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json().get("result", "No result found.")
            st.write(f"Answer: {result}")

            # User feedback (accuracy and satisfaction)
            st.subheader("Rate the response")
            accuracy = st.slider("How accurate was the response?", 0, 10, step=1)
            satisfaction = st.slider("How satisfied are you with the result?", 0, 10, step=1)

            # Display the ratings
            accuracy_placeholder.write(f"Accuracy Rating: {accuracy}/10")
            satisfaction_placeholder.write(f"User Satisfaction: {satisfaction}/10")

            # Optionally, send these ratings to your backend for logging or future analysis
            feedback_data = {"accuracy": accuracy, "satisfaction": satisfaction}
            feedback_response = requests.post(f"{API_URL}/feedback", json=feedback_data)

        else:
            st.error(f"Failed to retrieve answer. Status code: {response.status_code}")
    else:
        st.warning("Please enter a query.")
