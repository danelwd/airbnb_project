import sqlite3
import json
import pandas as pd
from datetime import datetime
import os

class ExplanationDatabase:
    """Database manager for storing and retrieving LLM explanations"""
    
    def __init__(self, db_path="llm_explanations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create explanations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                property_features TEXT,
                predicted_price REAL,
                prompt TEXT,
                response TEXT,
                model_used TEXT DEFAULT 'gpt-4',
                user_platform TEXT DEFAULT 'unknown',
                session_id TEXT
            )
        ''')
        
        # Create analytics table for insights
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_predictions INTEGER,
                avg_price REAL,
                most_common_room_type TEXT,
                avg_rating REAL,
                platform_usage TEXT
            )
        ''')
        
        # Create user_feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                explanation_id INTEGER,
                feedback_rating INTEGER,
                feedback_text TEXT,
                FOREIGN KEY (explanation_id) REFERENCES explanations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_explanation(self, features, price, prompt, response, platform="unknown", session_id=None):
        """Store a new explanation in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO explanations (property_features, predicted_price, prompt, response, user_platform, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (json.dumps(features), price, prompt, response, platform, session_id))
        
        explanation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return explanation_id
    
    def get_explanation_history(self, limit=10, platform=None):
        """Retrieve recent explanation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if platform:
            cursor.execute('''
                SELECT timestamp, property_features, predicted_price, response, user_platform
                FROM explanations 
                WHERE user_platform = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (platform, limit))
        else:
            cursor.execute('''
                SELECT timestamp, property_features, predicted_price, response, user_platform
                FROM explanations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for row in results:
            history.append({
                'timestamp': row[0],
                'features': json.loads(row[1]),
                'price': row[2],
                'explanation': row[3],
                'platform': row[4]
            })
        
        return history
    
    def get_analytics(self):
        """Get analytics about explanations and usage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(predicted_price) as avg_price,
                COUNT(DISTINCT user_platform) as platforms_used
            FROM explanations
        ''')
        
        stats = cursor.fetchone()
        
        # Get most common room type
        cursor.execute('''
            SELECT 
                json_extract(property_features, '$.room_type') as room_type,
                COUNT(*) as count
            FROM explanations
            GROUP BY room_type
            ORDER BY count DESC
            LIMIT 1
        ''')
        
        room_type_result = cursor.fetchone()
        most_common_room_type = room_type_result[0] if room_type_result else "Unknown"
        
        # Get platform usage
        cursor.execute('''
            SELECT user_platform, COUNT(*) as count
            FROM explanations
            GROUP BY user_platform
            ORDER BY count DESC
        ''')
        
        platform_usage = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_predictions': stats[0],
            'avg_price': stats[1],
            'platforms_used': stats[2],
            'most_common_room_type': most_common_room_type,
            'platform_usage': dict(platform_usage)
        }
    
    def store_feedback(self, explanation_id, rating, feedback_text=""):
        """Store user feedback for an explanation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (explanation_id, feedback_rating, feedback_text)
            VALUES (?, ?, ?)
        ''', (explanation_id, rating, feedback_text))
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                AVG(feedback_rating) as avg_rating,
                COUNT(*) as total_feedback,
                COUNT(CASE WHEN feedback_rating >= 4 THEN 1 END) as positive_feedback
            FROM user_feedback
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'avg_rating': stats[0],
            'total_feedback': stats[1],
            'positive_feedback': stats[2],
            'satisfaction_rate': (stats[2] / stats[1] * 100) if stats[1] > 0 else 0
        }
    
    def export_data(self, format='csv'):
        """Export explanation data for analysis"""
        conn = sqlite3.connect(self.db_path)
        
        if format == 'csv':
            # Export explanations
            explanations_df = pd.read_sql_query('''
                SELECT * FROM explanations
            ''', conn)
            
            # Export feedback
            feedback_df = pd.read_sql_query('''
                SELECT * FROM user_feedback
            ''', conn)
            
            conn.close()
            
            # Save to files
            explanations_df.to_csv('explanations_export.csv', index=False)
            feedback_df.to_csv('feedback_export.csv', index=False)
            
            return {
                'explanations_file': 'explanations_export.csv',
                'feedback_file': 'feedback_export.csv'
            }
        
        conn.close()
        return None
    
    def cleanup_old_data(self, days_old=30):
        """Clean up old explanation data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete explanations older than specified days
        cursor.execute('''
            DELETE FROM explanations 
            WHERE timestamp < datetime('now', '-{} days')
        '''.format(days_old))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count

# Example usage and testing
if __name__ == "__main__":
    db = ExplanationDatabase()
    
    # Test storing an explanation
    test_features = {
        'room_type': 'Entire home/apt',
        'bedrooms': 2,
        'bathrooms': 1.5,
        'accommodates': 4
    }
    
    explanation_id = db.store_explanation(
        features=test_features,
        price=150.0,
        prompt="Test prompt",
        response="Test explanation",
        platform="streamlit"
    )
    
    print(f"Stored explanation with ID: {explanation_id}")
    
    # Test retrieving history
    history = db.get_explanation_history(limit=5)
    print(f"Retrieved {len(history)} explanations")
    
    # Test analytics
    analytics = db.get_analytics()
    print(f"Analytics: {analytics}") 