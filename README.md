# 🏠 Airbnb Price Predictor Bot

An intelligent Telegram bot that predicts Airbnb property prices using machine learning and provides detailed explanations through LLM integration.

## 🚀 What This Project Does

This project creates a comprehensive Airbnb price prediction system that:
- **Analyzes property descriptions** from user messages
- **Predicts nightly prices** using trained machine learning models
- **Provides detailed explanations** for price predictions using LLM
- **Stores all interactions** in a SQLite database for analysis
- **Handles edge cases** and validates input data

## 🛠️ Features

### Core Functionality
- **Smart Message Recognition**: Identifies relevant property descriptions
- **Price Prediction**: Uses ML models to predict nightly rates
- **LLM Integration**: Generates detailed explanations for predictions
- **Data Validation**: Checks for realistic property values
- **Database Storage**: Saves all bot interactions for analysis

### Input Validation
- **Relevance Check**: Distinguishes between property descriptions and irrelevant messages
- **Value Validation**: Ensures bedroom/bathroom counts are realistic (1-20 bedrooms, 0.5-10 bathrooms)
- **Clear Error Messages**: Provides helpful guidance when input is invalid

## 📁 Project Structure

```
airbnb_new/
├── src/advisor/
│   ├── telegramBOT.py      # Main bot implementation
│   ├── database_manager.py  # Database operations
│   └── llm_explainer.py    # LLM integration
├── model/
│   ├── best_model.pkl      # Trained ML model
│   └── price_trained_model.pkl
├── notebooks/
│   ├── EDA.ipynb           # Data exploration
│   └── testme.ipynb        # Complete project workflow
├── data/
│   ├── listings.csv        # Raw Airbnb data
│   ├── data_cleaned.csv    # Cleaned data
│   └── my_processed_listings.csv # Feature-engineered data
└── requirements.txt         # Python dependencies
```

## 🔄 Complete Project Workflow

### **Phase 1: Data Processing & Model Development (Steps 1-6)**

#### **Step 1-2: Data Loading & Cleaning**
- Load raw Airbnb listings data (`listings.csv`)
- Select important features: room_type, bedrooms, bathrooms, price, host_is_superhost, etc.
- Clean price data (remove $ symbols, convert to numeric)
- Handle missing values and outliers
- Filter realistic price ranges ($40-$1000 per night)

#### **Step 3: Feature Engineering**
- Convert categorical variables using One-Hot Encoding
- Create boolean flags for room types and superhost status
- Prepare features for ML model training

#### **Step 4: Data Splitting**
- Split data into training (80%) and testing (20%) sets
- Save split datasets for model training and evaluation

#### **Step 5: Model Training**
- Train CatBoost Regressor model
- Optimize hyperparameters (iterations=100, learning_rate=0.1, depth=6)
- Save trained model as `price_trained_model.pkl`

#### **Step 6: Streamlit Web Interface**
- Create user-friendly web form for property input
- Integrate trained model for real-time predictions
- Display price predictions with visualizations

### **Phase 2: Telegram Bot & LLM Integration (Steps 7-8)**

#### **Step 7: Telegram Bot Implementation**
- Create bot using python-telegram-bot library
- Implement message parsing and property information extraction
- Integrate ML model for price predictions
- Add interactive buttons and commands

#### **Step 8: LLM Integration**
- Connect to OpenAI API for detailed explanations
- Generate contextual prompts for price reasoning
- Store interactions in SQLite database
- Implement fallback explanations when LLM is unavailable

### **Phase 3: Testing & Documentation (Steps 9-10)**

#### **Step 9: Edge Case Testing**
- Test with invalid inputs and edge cases
- Validate property value ranges
- Handle irrelevant messages gracefully

#### **Step 10: Documentation & Deployment**
- Create comprehensive README
- Prepare for GitHub upload
- Document all features and usage

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Telegram Bot Token (from @BotFather)
- OpenAI API Key (for LLM explanations)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/airbnb-price-predictor.git
cd airbnb-price-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
TELEGRAM_BOT_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_key_here
```

### Running Different Components

#### **Option 1: Run the Telegram Bot**
```bash
cd src/advisor
python telegramBOT.py
```

#### **Option 2: Run the Streamlit Web Interface**
```bash
cd src/advisor
streamlit run streamlit_app.py
```

#### **Option 3: Run Complete Data Pipeline**
```bash
cd notebooks
jupyter notebook testme.ipynb
# Execute all cells in order
```

## 💬 How to Use the Bot

### Basic Commands
- `/start` - Welcome message and instructions
- `/help` - Detailed help information
- `/example` - Example property descriptions

### Sending Property Descriptions

**Format**: Describe your property with key details

**Examples**:
```
✅ Good examples:
"2 bedroom apartment with 1 bathroom, accommodates 4 guests"
"3 bedroom house with 2 bathrooms, superhost, 4.9 stars"
"1 bedroom studio with 1 bathroom for 2 people"

❌ Invalid examples:
"hello world" (not a property)
"100 bedrooms" (unrealistic value)
"0 bedroom apartment" (invalid count)
```

### What the Bot Returns

**Successful Prediction**:
```
💰 Price Prediction: $280.08 per night

Property Details:
• Room Type: Entire home/apt
• Bedrooms: 2
• Bathrooms: 1.0
• Accommodates: 4 guests

Explanation: [Detailed LLM explanation]

Note: This is an estimate based on similar properties.
```

**Error Messages**:
```
❌ Irrelevant message
Please send a property description with details like:
• Number of bedrooms
• Number of bathrooms
• Number of guests
• Room type (entire home, private room, shared room)

Example: "2 bedroom apartment with 1 bathroom, accommodates 4 guests"
```

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm**: CatBoost Regressor
- **Features**: Bedrooms, bathrooms, room type, superhost status, review ratings
- **Training Data**: Airbnb listings with price information
- **Model Files**: `best_model.pkl`, `price_trained_model.pkl`
- **Performance**: R² Score: 0.53, MAE: 67.71

### Data Processing Pipeline
- **Raw Data**: 7,804 listings with 79 features
- **Cleaned Data**: 4,464 listings after filtering and cleaning
- **Feature Engineering**: One-hot encoding for categorical variables
- **Data Split**: 3,571 training samples, 893 test samples

### LLM Integration
- **Provider**: OpenAI GPT models
- **Purpose**: Generate detailed price explanations
- **Fallback**: Simple explanations when LLM is unavailable

### Database Schema
```sql
CREATE TABLE bot_interactions (
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
);
```

## 🧪 Testing

### Edge Case Testing
The bot has been tested with:
- **Irrelevant messages**: "hello world", "a", "123"
- **Invalid values**: "0 bedrooms", "-1 bathrooms", "100 bedrooms"
- **Extreme values**: "0.5 bedrooms", "15 bathrooms"
- **Valid inputs**: Various property descriptions

### Expected Behavior
- ✅ **Relevant messages**: Returns price predictions
- ❌ **Irrelevant messages**: Returns error with examples
- ⚠️ **Invalid values**: Returns warnings with recommendations
- 💾 **All interactions**: Stored in database for analysis

## 📊 Performance

### Response Time
- **Message processing**: < 2 seconds
- **Price prediction**: < 1 second
- **LLM explanation**: 2-5 seconds (depending on OpenAI API)

### Accuracy
- **Input validation**: 100% (catches all invalid inputs)
- **Price prediction**: Based on trained model accuracy
- **Error handling**: Comprehensive coverage of edge cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:
1. Check the error messages in the terminal
2. Verify your environment variables
3. Ensure all dependencies are installed
4. Check the database connection

## 🎯 Future Improvements

- [ ] Add location-based pricing
- [ ] Integrate with real-time market data
- [ ] Add image recognition for property photos
- [ ] Implement user preferences and history
- [ ] Add multi-language support
- [ ] Improve model accuracy with more features
- [ ] Add A/B testing for different ML algorithms
- [ ] Implement caching for faster responses
