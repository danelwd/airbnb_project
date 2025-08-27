import os
import logging
import joblib
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from dotenv import load_dotenv
import re
import json
from datetime import datetime
from .llm_explainer import LLMExplainer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AirbnbPriceBot:
    def __init__(self):
        """Initialize the Telegram bot with model and LLM integration"""
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            # Fallback to hardcoded token if environment variable not found
            self.token = "7483873824:AAE3IoRdWvRWJeMcJ8XkmD75oVqiNjR4c68"
        
        # Load the trained model
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'price_trained_model.pkl')
        try:
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a simple model for testing
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor()
            # Train on simple data that matches our features
            X_simple = [
                [1, 1.0, 1, 10, 80.0, 80.0, 0, False, False, False, True],
                [2, 1.0, 1, 10, 80.0, 80.0, 0, False, False, False, True],
                [3, 2.0, 1, 10, 80.0, 80.0, 0, False, False, False, True],
                [1, 1.0, 1, 10, 90.0, 90.0, 0, False, False, False, False],
                [2, 2.0, 1, 10, 90.0, 90.0, 0, False, False, False, False]
            ]
            y_simple = [80, 120, 180, 100, 150]  # Simple price predictions
            self.model.fit(X_simple, y_simple)
            logger.info("Using simple trained model for testing")
        
        # Load training data for feature alignment
        try:
            self.X_train = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'X_train.csv'))
            logger.info("Training data loaded for feature alignment")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            self.X_train = None
        
        # Initialize conversation states
        self.user_states = {}
        
        # Initialize LLM explainer
        try:
            self.explainer = LLMExplainer()
        except Exception as e:
            logger.error(f"Failed to initialize LLM explainer: {e}")
            self.explainer = None
        
       #חיבור לדאטהבייס
        # Initialize database connection
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(__file__), 'llm_explanations.db')
            self.conn = sqlite3.connect(db_path)
            logger.info("Database connection established successfully")
            # Create interactions table
            self.create_interactions_table()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.conn = None
        
    def extract_property_info(self, text: str) -> dict:
        """Extract property information from user text using regex patterns"""
        info = {}
        
        # Room type patterns
        room_patterns = {
            'Entire home/apt': r'\b(entire|full|whole)\s+(home|apartment|apt|house|unit)\b',
            'Private room': r'\b(private|separate)\s+room\b',
            'Shared room': r'\b(shared|common)\s+room\b'
        }
        
        for room_type, pattern in room_patterns.items():
            if re.search(pattern, text.lower()):
                info['room_type'] = room_type
                break
        
        # If no room type found, default to entire home
        if 'room_type' not in info:
            if 'apartment' in text.lower() or 'home' in text.lower():
                info['room_type'] = 'Entire home/apt'
            else:
                info['room_type'] = 'Entire home/apt'
        
        # Number patterns
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        
        # Bedrooms
        bedroom_match = re.search(r'\b(\d+)\s+(bedroom|bedrooms)\b', text.lower())
        if bedroom_match:
            info['bedrooms'] = int(bedroom_match.group(1))
        else:
            # Default to 1 bedroom if not specified
            info['bedrooms'] = 1
        
        # Bathrooms
        bathroom_match = re.search(r'\b(\d+(?:\.\d+)?)\s+(bathroom|bathrooms)\b', text.lower())
        if bathroom_match:
            info['bathrooms'] = float(bathroom_match.group(1))
        else:
            # Default to 1 bathroom if not specified
            info['bathrooms'] = 1.0
        
        # Accommodates (guests)
        guest_match = re.search(r'\b(\d+)\s+(guest|guests|people|person)\b', text.lower())
        if guest_match:
            info['accommodates'] = int(guest_match.group(1))
        else:
            # Default to 2 guests if not specified
            info['accommodates'] = 2
        
        # Superhost
        if re.search(r'\b(superhost|super host)\b', text.lower()):
            info['host_is_superhost'] = True
        else:
            info['host_is_superhost'] = False
        
        # Review scores
        review_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:star|stars|rating|score)\b', text.lower())
        if review_match:
            info['review_scores_rating'] = float(review_match.group(1))
        else:
            info['review_scores_rating'] = 80.0
        
        # Availability
        avail_match = re.search(r'\b(\d+)\s+(day|days)\s+(available|availability)\b', text.lower())
        if avail_match:
            info['availability_30'] = int(avail_match.group(1))
        else:
            info['availability_30'] = 10
        
        return info
    
    def generate_better_prompt(self, property_info: dict, predicted_price: float) -> str:
        """Generate a better prompt for LLM explanation"""
        prompt = f"""
        Property Details:
        - Room Type: {property_info.get('room_type', 'Entire home/apt')}
        - Bedrooms: {property_info.get('bedrooms', 1)}
        - Bathrooms: {property_info.get('bathrooms', 1.0)}
        - Accommodates: {property_info.get('accommodates', 2)} guests
        - Superhost: {property_info.get('host_is_superhost', False)}
        - Review Rating: {property_info.get('review_scores_rating', 80.0)} stars
        
        Predicted Price: ${predicted_price:.2f} per night
        
        Please explain why this property costs ${predicted_price:.2f} per night. 
        Consider factors like room type, size, amenities, and market conditions.
        Give a clear, simple explanation in English.
        """
        return prompt

    def generate_simple_explanation(self, property_info: dict, predicted_price: float) -> str:
        """Generate simple explanation when LLM is not available"""
        explanation = f"""
        The price of ${predicted_price:.2f} per night is based on:
        • Room Type: {property_info.get('room_type', 'Entire home/apt')} (typically commands higher prices)
        • Size: {property_info.get('bedrooms', 1)} bedroom(s) and {property_info.get('bathrooms', 1.0)} bathroom(s)
        • Capacity: Accommodates {property_info.get('accommodates', 2)} guests
        • Quality: {'Superhost property' if property_info.get('host_is_superhost', False) else 'Standard host'}
        • Reviews: {property_info.get('review_scores_rating', 80.0)} star rating
        """
        return explanation
    #מזהה האם ההודעה רלוונטית לתיאור נכס
    def is_relevant_message(self, text: str) -> bool:
        """Check if message is relevant to property description"""
        relevant_keywords = [
            'bedroom', 'bathroom', 'apartment', 'house', 'room',
            'home', 'guest', 'people', 'superhost', 'star'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in relevant_keywords)
            #האם הערכים הגיוניים

    def validate_property_values(self, property_info: dict) -> tuple[bool, str]:
        """Validate if property values are realistic"""
        warnings = []
        # Check bedrooms (1-20 is realistic)
        bedrooms = property_info.get('bedrooms', 1)
        if bedrooms < 1:
            warnings.append("Bedrooms cannot be less than 1")
        elif bedrooms > 20:
            warnings.append(f"{bedrooms} bedrooms seems unrealistic (recommended: 1-20)")
        
        # Check bathrooms (0.5-10 is realistic)
        bathrooms = property_info.get('bathrooms', 1.0)
        if bathrooms < 0.5:
            warnings.append("Bathrooms cannot be less than 0.5")
        elif bathrooms > 10:
            warnings.append(f"{bathrooms} bathrooms seems unrealistic (recommended: 0.5-10)")
        
        # Check accommodates (1-20 is realistic)
        accommodates = property_info.get('accommodates', 2)
        if accommodates < 1:
            warnings.append("Accommodates cannot be less than 1")
        elif accommodates > 20:
            warnings.append(f"{accommodates} guests seems unrealistic (recommended: 1-20)")
        
        if warnings:
            return False, " | ".join(warnings)
        return True, ""

    def get_error_message(self, text: str, is_relevant: bool, validation_result: tuple) -> str:
        """Generate appropriate error message based on message type and validation"""
        if not is_relevant:
            return """❌ Irrelevant message

Please send a property description with details like:
• Number of bedrooms
• Number of bathrooms
• Number of guests
• Room type (entire home, private room, shared room)

Example: "2 bedroom apartment with 1 bathroom, accommodates 4 guests" """
        
        is_valid, warnings = validation_result
        if not is_valid:
            return f"""⚠️ Invalid property values

{warnings}

Please check your data and try again."""
        
        return ""
    
    def create_interactions_table(self):
        """Create the bot interactions table if it doesn't exist"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER,
                    user_message TEXT,
                    predicted_price REAL,
                    property_details TEXT,
                    llm_prompt TEXT,
                    llm_response TEXT,
                    room_type TEXT,
                    bedrooms INTEGER,
                    bathrooms REAL,
                    accommodates INTEGER,
                    superhost BOOLEAN,
                    review_rating REAL
                )
            """)
            self.conn.commit()
            logger.info("Bot interactions table created successfully")
        except Exception as e:
            logger.error(f"Failed to create interactions table: {e}")

    def save_interaction(self, user_id: int, user_message: str, predicted_price: float, 
                        property_info: dict, llm_prompt: str, llm_response: str):
        """Save bot interaction to database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO bot_interactions 
                (user_id, user_message, predicted_price, property_details, llm_prompt, llm_response,
                 room_type, bedrooms, bathrooms, accommodates, superhost, review_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
                user_id,
                user_message,
                predicted_price,
                str(property_info),
                llm_prompt,
                llm_response,
                property_info.get('room_type'),
                property_info.get('bedrooms'),
                property_info.get('bathrooms'),
                property_info.get('accommodates'),
                property_info.get('host_is_superhost'),
                property_info.get('review_scores_rating')
            ))
            self.conn.commit()
            logger.info(f"Interaction saved for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
    
    def prepare_features(self, property_info: dict) -> pd.DataFrame:
        """Prepare features for model prediction"""
        # Create DataFrame with the exact columns the model expects
        features = {
            'bedrooms': property_info.get('bedrooms', 1),
            'bathrooms': property_info.get('bathrooms', 1.0),
            'host_listings_count': property_info.get('host_listings_count', 1),
            'availability_30': property_info.get('availability_30', 10),
            'review_scores_rating': property_info.get('review_scores_rating', 80.0),
            'review_scores_location': property_info.get('review_scores_location', 80.0),
            'calculated_host_listings_count_private_rooms': property_info.get('calculated_host_listings_count_private_rooms', 0),
            'room_type_Hotel room': False,
            'room_type_Private room': False,
            'room_type_Shared room': False,
            'host_is_superhost_t': property_info.get('host_is_superhost', False)
        }
        
        # Set room type based on property_info
        room_type = property_info.get('room_type', 'Entire home/apt')
        if room_type == 'Hotel room':
            features['room_type_Hotel room'] = True
        elif room_type == 'Private room':
            features['room_type_Private room'] = True
        elif room_type == 'Shared room':
            features['room_type_Shared room'] = True
        
        # Convert boolean to proper format
        features['host_is_superhost_t'] = bool(features['host_is_superhost_t'])
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def predict_price(self, property_info: dict) -> tuple:
        """Predict price and generate explanation"""
        try:
            # Prepare features
            features = self.prepare_features(property_info)
            
            # Make prediction
            predicted_price = self.model.predict(features)[0]
            
            # Generate explanation using LLM if available
            if self.explainer:
                try:
                    # Use improved prompt
                    prompt = self.generate_better_prompt(property_info, predicted_price)
                    explanation = self.explainer.generate_llm_explanation(property_info, predicted_price)
                    logger.info("LLM explanation generated successfully")
                except Exception as e:
                    logger.error(f"LLM explanation failed: {e}")
                    explanation = self.generate_simple_explanation(property_info, predicted_price)
            else:
                explanation = self.generate_simple_explanation(property_info, predicted_price)
            
            return predicted_price, explanation
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, f"Error making prediction: {str(e)}"
    

    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        welcome_message = """
🏠 *Welcome to Airbnb Price Predictor Bot!*

I can help you predict the price of Airbnb properties based on their features.

*How to use:*
1. Send me property details as text
2. Include information like:
   - Room type (entire home, private room, shared room)
   - Number of bedrooms and bathrooms
   - Number of guests it accommodates
   - Whether it's a superhost property
   - Review ratings

*Example:* "2 bedroom apartment with 1 bathroom, accommodates 4 guests, superhost, 4.8 star rating"

*Commands:*
/start - Show this message
/help - Get help
/example - See example property descriptions

Ready to predict some prices! 🚀
        """
        
        keyboard = [
            [InlineKeyboardButton("📝 Send Property Details", callback_data='send_details')],
            [InlineKeyboardButton("❓ How to use", callback_data='how_to_use')],
            [InlineKeyboardButton("💡 Example", callback_data='example')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        if update.message is None:
            return
            
        help_text = """
*🤖 Airbnb Price Predictor Bot Help*

*What I can do:*
• Predict Airbnb property prices based on features
• Provide explanations for predictions
• Handle various property types and configurations

*Required information:*
• Room type (entire home/apt, private room, shared room)
• Number of bedrooms
• Number of bathrooms
• Number of guests accommodated

*Optional information:*
• Superhost status
• Review ratings
• Availability

*Tips for better predictions:*
• Be specific about room type
• Include all bedroom/bathroom counts
• Mention if it's a superhost property
• Include review scores if available

*Example:* "3 bedroom entire apartment with 2 bathrooms, accommodates 6 guests, superhost, 4.9 stars"

Need more help? Just ask! 😊
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def example_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /example command"""
        if update.message is None:
            return
            
        examples = """
*📋 Example Property Descriptions:*

*Example 1:*
"2 bedroom entire apartment with 1 bathroom, accommodates 4 guests, superhost, 4.8 star rating"

*Example 2:*
"Private room with 1 bedroom and shared bathroom, accommodates 2 guests, 4.5 stars"

*Example 3:*
"1 bedroom entire home with 1 bathroom, accommodates 2 guests, superhost, 4.9 rating"

*Example 4:*
"Shared room with 2 beds, accommodates 2 guests, 4.2 stars"

Try copying one of these examples and modifying the details! 🎯
        """
        
        await update.message.reply_text(examples, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages"""
        if update.message is None or update.message.text is None:
            return
            
        user_id = update.message.from_user.id
        text = update.message.text
        
        # Check if message is relevant
        is_relevant = self.is_relevant_message(text)
        
        if not is_relevant:
            error_msg = self.get_error_message(text, is_relevant, (True, ""))
            await update.message.reply_text(error_msg)
            return
        
        # Extract property information
        property_info = self.extract_property_info(text)
        
        # Validate property values
        validation_result = self.validate_property_values(property_info)
        is_valid, warnings = validation_result
        
        if not is_valid:
            error_msg = self.get_error_message(text, is_relevant, validation_result)
            await update.message.reply_text(error_msg)
            return
        
        # Show processing message
        processing_msg = await update.message.reply_text("🔍 Analyzing property details...")
        
        try:
            # Make prediction
            predicted_price, explanation = self.predict_price(property_info)
            
            if predicted_price is None:
                await processing_msg.edit_text(f"❌ {explanation}")
                return
            
            # Format response
            response = f"""
💰 *Price Prediction: ${predicted_price:.2f} per night*

*Property Details:*
• Room Type: {property_info.get('room_type', 'Entire home/apt')}
• Bedrooms: {property_info.get('bedrooms', 1)}
• Bathrooms: {property_info.get('bathrooms', 1.0)}
• Accommodates: {property_info.get('accommodates', 2)} guests
{f"• Superhost: ✅" if property_info.get('host_is_superhost', False) else ""}
{f"• Review Rating: {property_info.get('review_scores_rating', 80.0)} stars" if 'review_scores_rating' in property_info else ""}

*Explanation:*
{explanation}

*Note:* This is an estimate based on similar properties. Actual prices may vary.
            """
            
            await processing_msg.edit_text(response, parse_mode='Markdown')
            
            # Save interaction to database
            if self.conn:
                try:
                    llm_prompt = self.generate_better_prompt(property_info, predicted_price)
                    self.save_interaction(
                        user_id=user_id,
                        user_message=text,
                        predicted_price=predicted_price,
                        property_info=property_info,
                        llm_prompt=llm_prompt,
                        llm_response=explanation
                    )
                    logger.info(f"Interaction saved for user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to save interaction: {e}")
            else:
                logger.warning("Database not available, skipping interaction save")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await processing_msg.edit_text(
                "❌ Sorry, I encountered an error while processing your request. "
                "Please try again with different property details."
            )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks"""
        query = update.callback_query
        if update.callback_query is None:
            return

        query = update.callback_query
        await query.answer()

        if getattr(query, "data", None) == 'send_details':
            await query.edit_message_text(
                "📝 Please send me the property details as text.\n\n"
                "Include information like room type, bedrooms, bathrooms, etc.\n\n"
                "Example: '2 bedroom entire apartment with 1 bathroom, accommodates 4 guests'"
            )
        
        
        elif query.data == 'how_to_use':
            await self.help_command(update, context)
        
        elif query.data == 'example':
            await self.example_command(update, context)
    
    def run(self):
        """Run the bot"""
        # Create application
        application = Application.builder().token(self.token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("example", self.example_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Start the bot
        logger.info("Starting Airbnb Price Predictor Bot...")
        try:
            application.run_polling()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            # Cleanup
            if hasattr(self, 'explainer'):
                self.explainer.close()

if __name__ == "__main__":
    bot = AirbnbPriceBot()
    bot.run()