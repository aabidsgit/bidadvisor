import os
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable

class BidAdvisorApp:
    def _init_(self):
        load_dotenv()
        self.regressor = None
        self.classifier = None
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_bid_path = os.path.join(self.model_dir, "model_bid_predictor.pkl")
        self.model_win_path = os.path.join(self.model_dir, "model_win_predictor.pkl")
        self._load_or_train_models()
        self._setup_llm()

    def _load_or_train_models(self):
        if os.path.exists(self.model_bid_path) and os.path.exists(self.model_win_path):
            print("📦 Loading saved models...")
            self.regressor = joblib.load(self.model_bid_path)
            self.classifier = joblib.load(self.model_win_path)
        else:
            print("🔧 Training models from scratch...")
            df = pd.read_csv("data/auction_history.csv")
            df = df[df['winning_bid'] > 0]

            features = [
                'location', 'property_type', 'bedrooms', 'bathrooms',
                'sqft', 'year_built', 'min_bid', 'bidders_count', 'auction_month'
            ]

            X = df[features]
            y_bid = df['winning_bid']
            y_status = df['status'].map({'Won': 1, 'Lost': 0})

            X_train, _, y_bid_train, _, y_status_train, _ = train_test_split(
                X, y_bid, y_status, test_size=0.2, random_state=42
            )

            cat_cols = ['location', 'property_type']
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ], remainder='passthrough')

            self.regressor = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            self.classifier = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])

            self.regressor.fit(X_train, y_bid_train)
            self.classifier.fit(X_train, y_status_train)

            joblib.dump(self.regressor, self.model_bid_path)
            joblib.dump(self.classifier, self.model_win_path)
            print("✅ Models trained and saved.")

    def _setup_llm(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        self.prompt_template = PromptTemplate.from_template(
    """
    You are a smart real estate assistant.

    Extract the following fields from the input text and ensure no field is missing. 
    If the user has not provided a value, you MUST guess a realistic value based on common knowledge or defaults. 
    Return values in proper data types and DO NOT leave anything null or blank.

    Required fields:
    - location (e.g., Dallas)
    - property_type (Residential or Commercial)
    - bedrooms (int)
    - bathrooms (float)
    - sqft (int)
    - year_built (int)
    - min_bid (float)
    - bidders_count (int)
    - auction_month (int, guess from current month or context)

    Format output strictly as JSON.

    Input: {user_input}
    """
)

        
        self.chain: Runnable = self.prompt_template | self.llm

    def parse_input(self, user_input: str) -> dict:
        response = self.chain.invoke({"user_input": user_input})
        response_text = response.text().strip()

        # Remove Markdown formatting if present
        if response_text.startswith("json"):
            response_text = response_text.replace("json", "").replace("```", "").strip()

        print("🧪 Cleaned LLM response:\n", response_text)
        parsed = json.loads(response_text)

        # Add fallback defaults
        

        return parsed

    def predict(self, parsed_input: dict):
        df_input = pd.DataFrame([parsed_input])
        predicted_bid = self.regressor.predict(df_input)[0]
        win_prob = self.classifier.predict_proba(df_input)[0][1]
        return predicted_bid, win_prob

    def run(self, user_input: str):
        parsed = self.parse_input(user_input)
        bid, win_chance = self.predict(parsed)

        print(f"📊 Suggested Bid: ${bid:,.0f}")
        print(f"✅ Win Probability: {win_chance * 100:.2f}%")

# --- Run the app ---
if _name_ == "_main_":
    app = BidAdvisorApp()
    user_query = "Suggest a bid for a 4BHK in Tampa, and 8 months in auction,4000 sqft, built recently with a min bid of 100000"
    app.run(user_query)
