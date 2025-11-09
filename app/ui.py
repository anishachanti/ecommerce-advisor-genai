import requests
import streamlit as st

st.title("🛍️ E-Commerce Shopping Advisor")

API_URL = "http://localhost:8000/chat"

user_query = st.text_input("Ask something like: 'Suggest a phone under 20K with good camera'")

if st.button("Ask Advisor"):
    if user_query.strip():
        try:
            response = requests.get(API_URL, params={"q": user_query})
            
            if response.status_code == 200:
                data = response.json()
                st.markdown("### 🤖 Advisor Response:")
                st.write(data["answer"])
                

            else:
                st.error("⚠️ Error from backend: " + str(response.text))

        except Exception as e:
            st.error("❌ Could not connect to backend: " + str(e))
