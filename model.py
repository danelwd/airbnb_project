## 19

import pandas as pd
from sklearn.model_selection import train_test_split

# שלב 1: קריאת הנתונים
df = pd.read_csv("C:/Users/Danel Wittner/Desktop/ai/airbnb_new/notebooks/data/my_processed_listings.csv")  # type: ignore # טוען את הנתונים מקובץ CSV בתיקיית data

# שלב 2: הפרדת משתני קלט (features) ממשתנה המטרה (price)
# כל העמודות חוץ מ-"price" הן משתני קלט
X = df.drop("price", axis=1)
# עמודת "price" היא משתנה המטרה
y = df["price"]

# שלב 3: חלוקת הנתונים לסט אימון (Train) וסט בדיקה (Test)
# 80% מהנתונים לאימון, 20% לבדיקה
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# random_state קובע את ה"גרעין" להגרלה, כדי שהתוצאה תהיה שחזורית

#  אושלב 4: שמירת הסטים לקבצים (אם נרצה להעריך את המודל בעתיד עלתם נתונים)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# הסברים:
# - הפרדנו בין משתני קלט למשתנה מטרה כדי שהמודל ילמד לחזות את המחיר על סמך מאפייני הנכס בלבד.
# - חילקנו את הנתונים לסט אימון וסט בדיקה כדי שנוכל להעריך את המודל על נתונים שהוא לא ראה בתהליך האימון.
# - שמרנו את הסטים לקבצים כדי שנוכל לחזור אליהם בעתיד ולהשוות ביצועי מודלים שונים על אותם נתונים.

print("✅ Data split completed. Shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)



## 22

import pandas as pd
from catboost import CatBoostRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# Step 1: Load preprocessed data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Step 1.5: Identify and handle categorical features
# Get categorical columns (object/string columns)
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"📊 Categorical features found: {categorical_features}")

# Convert categorical features to category type for CatBoost
for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Get the indices of categorical features
cat_features_indices = [X_train.columns.get_loc(col) for col in categorical_features]
print(f"📊 Categorical feature indices: {cat_features_indices}")

# Step 2: Initialize the CatBoost model
model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=0  # שים 100 אם אתה רוצה לראות עדכונים כל 100 איטרציות
)

# Step 3: Train the model with categorical features
model.fit(X_train, y_train.values.ravel(), cat_features=cat_features_indices)

# Step 4: Evaluate on the test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Evaluation Results:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 5: Save model to .pkl file
#joblib.dump(model, "models/price_predictor_catboost.pkl")
#print("💾 Model saved to models/price_predictor_catboost.pkl")

# Step 6: Interpretation of results
if mae < 30:
    print("✅ The model is highly accurate with very small price errors.")
elif mae < 80:
    print("⚠️ The model is decent, but may over- or under-predict prices by $50–100 on average.")
else:
    print("❌ The model is inaccurate — some predictions may be way off.")

if r2 > 0.8:
    print("✅ The model explains most of the price variability — strong performance.")
elif r2 > 0.5:
    print("⚠️ Moderate performance — some important factors might be missing.")
else:
    print("❌ The model performs poorly — features may be insufficient.")


## 22
