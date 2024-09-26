import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"  # Make sure this matches the FastAPI server

# Streamlit UI components
st.title("Document QA System")

# File upload section
st.header("Upload PDF File")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Display the filename
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Upload file to the FastAPI backend
    files = {"uploaded_file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post(f"{API_URL}/upload/file", files=files)

    if response.status_code == 200:
        st.success("File successfully uploaded and ingested.")
    else:
        st.error(f"Failed to upload file. Status code: {response.status_code}")

# Query section
st.header("Query the Document")
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query:
        # Send the query to FastAPI backend
        params = {"query": query}
        response = requests.get(f"{API_URL}/query", params=params)
        print(response)
        if response.status_code == 200:
            result = response.json().get("result", "No result found.")
            st.write(f"Answer: {result}")
        else:
            st.error(f"Failed to retrieve answer. Status code: {response.status_code}")
    else:
        st.warning("Please enter a query.")

