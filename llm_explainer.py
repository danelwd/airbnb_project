import os
import openai
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
import json

# Load environment variables
load_dotenv()

class LLMExplainer:
    """Enhanced LLM explainer for price predictions"""
    
    def __init__(self):
        """Initialize the LLM explainer"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
            self.use_llm = True
        else:
            self.use_llm = False
            print("⚠️ OpenAI API key not found. Using basic explanations.")
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing explanations"""
        try:
            self.conn = sqlite3.connect('llm_explanations.db')
            self.cursor = self.conn.cursor()
            
            # Create table if not exists
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS explanations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    property_info TEXT,
                    predicted_price REAL,
                    explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def generate_llm_explanation(self, property_info: dict, predicted_price: float) -> str:
        """Generate explanation using OpenAI LLM"""
        if not self.use_llm:
            return self.generate_basic_explanation(property_info, predicted_price)
        
        try:
            # Prepare property description
            property_desc = self.format_property_description(property_info)
            
            prompt = f"""
You are an expert Airbnb pricing analyst. Analyze this property and explain why the predicted price of ${predicted_price:.2f} per night is reasonable.

Property Details:
{property_desc}

Please provide a concise, professional explanation (2-3 sentences) that covers:
1. Key factors influencing the price
2. Market positioning (budget/mid-range/premium)
3. Value proposition for guests

Focus on the most important factors and keep the explanation clear and helpful.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Airbnb pricing expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            
            # Store in database
            self.store_explanation(property_info, predicted_price, explanation)
            
            return explanation
            
        except Exception as e:
            print(f"LLM explanation error: {e}")
            return self.generate_basic_explanation(property_info, predicted_price)
    
    def generate_basic_explanation(self, property_info: dict, predicted_price: float) -> str:
        """Generate basic explanation without LLM"""
        explanations = []
        
        # Room type explanation
        room_type = property_info.get('room_type', 'Entire home/apt')
        if room_type == 'Entire home/apt':
            explanations.append("Entire home/apartment typically commands higher prices due to privacy and full amenities")
        elif room_type == 'Private room':
            explanations.append("Private room offers good value for solo travelers or couples")
        else:
            explanations.append("Shared room is the most budget-friendly option for cost-conscious travelers")
        
        # Bedrooms explanation
        bedrooms = property_info.get('bedrooms', 1)
        if bedrooms > 2:
            explanations.append(f"{bedrooms} bedrooms support larger groups and families")
        elif bedrooms == 1:
            explanations.append("Single bedroom perfect for couples or solo travelers")
        
        # Bathrooms explanation
        bathrooms = property_info.get('bathrooms', 1.0)
        if bathrooms > 1:
            explanations.append(f"{bathrooms} bathrooms add convenience for groups")
        
        # Superhost explanation
        if property_info.get('host_is_superhost', False):
            explanations.append("Superhost status indicates high-quality experience and justifies premium pricing")
        
        # Review score explanation
        review_score = property_info.get('review_scores_rating', 80.0)
        if review_score >= 90:
            explanations.append("Excellent reviews justify premium pricing")
        elif review_score >= 80:
            explanations.append("Good reviews support competitive pricing")
        
        # Price range explanation
        if predicted_price < 100:
            explanations.append("Budget-friendly pricing for cost-conscious travelers")
        elif predicted_price < 200:
            explanations.append("Mid-range pricing for comfortable stays")
        else:
            explanations.append("Premium pricing for luxury accommodations")
        
        return " | ".join(explanations)
    
    def format_property_description(self, property_info: dict) -> str:
        """Format property information for LLM prompt"""
        desc_parts = []
        
        if 'room_type' in property_info:
            desc_parts.append(f"Room Type: {property_info['room_type']}")
        
        if 'bedrooms' in property_info:
            desc_parts.append(f"Bedrooms: {property_info['bedrooms']}")
        
        if 'bathrooms' in property_info:
            desc_parts.append(f"Bathrooms: {property_info['bathrooms']}")
        
        if 'accommodates' in property_info:
            desc_parts.append(f"Accommodates: {property_info['accommodates']} guests")
        
        if property_info.get('host_is_superhost', False):
            desc_parts.append("Superhost: Yes")
        
        if 'review_scores_rating' in property_info:
            desc_parts.append(f"Review Rating: {property_info['review_scores_rating']} stars")
        
        return "\n".join(desc_parts)
    
    def store_explanation(self, property_info: dict, predicted_price: float, explanation: str):
        """Store explanation in database"""
        try:
            self.cursor.execute('''
                INSERT INTO explanations (property_info, predicted_price, explanation)
                VALUES (?, ?, ?)
            ''', (json.dumps(property_info), predicted_price, explanation))
            self.conn.commit()
        except Exception as e:
            print(f"Database storage error: {e}")
    
    def get_explanation_history(self, limit: int = 5) -> list:
        """Get recent explanation history"""
        try:
            self.cursor.execute('''
                SELECT property_info, predicted_price, explanation, created_at
                FROM explanations
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'features': json.loads(row[0]),
                    'price': row[1],
                    'explanation': row[2],
                    'created_at': row[3]
                })
            
            return results
        except Exception as e:
            print(f"Database retrieval error: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close() 