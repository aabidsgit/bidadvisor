import streamlit as st
from bidadvisor_standalone import BidAdvisorApp  # assuming it's in the same folder

# Initialize app
advisor = BidAdvisorApp()

# Streamlit UI
st.set_page_config(page_title="BidAdvisor", layout="centered")
st.title("💼 BidAdvisor")

user_input = st.text_area("Describe your property to get a bid suggestion:", height=150)

if st.button("🔍 Predict Bid"):
    if user_input.strip():
        with st.spinner("Analyzing and Predicting..."):
            parsed = advisor.parse_input(user_input)
            bid, win_prob = advisor.predict(parsed)

        # Output
        st.success("Prediction Complete!")
        st.markdown(f"### 📊 Suggested Bid: ${bid:,.0f}")
        st.markdown(f"### ✅ Win Probability: {win_prob * 100:.2f}%")
    else:
        st.warning("Please enter a valid description.")
