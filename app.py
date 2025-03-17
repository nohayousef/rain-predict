# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import os

# # # ✅ Set page config
# # st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# # # Sidebar Navigation
# # st.sidebar.title("Navigation")
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# # # ✅ Load the dataset
# # @st.cache_data
# # def load_data():
# #     file_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\weatherAUS.csv"
# #     if not os.path.exists(file_path):
# #         st.error("❌ Dataset file not found! Check the file path.")
# #         return None
# #     df = pd.read_csv(file_path, parse_dates=["Date"])
# #     df.dropna(subset=["WindGustDir", "RainTomorrow", "Location", "Rainfall", 
# #                       "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"], inplace=True)
# #     df["Season"] = df["Date"].dt.month.map(
# #         lambda m: "Winter" if m in [12, 1, 2] else 
# #                   "Spring" if m in [3, 4, 5] else 
# #                   "Summer" if m in [6, 7, 8] else 
# #                   "Autumn"
# #     )
# #     return df

# # df = load_data()

# # # ✅ Check for Model Availability
# # model_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting2_pipeline.pkl"
# # model = None

# # if os.path.exists(model_path):
# #     try:
# #         import joblib
# #         model = joblib.load(model_path)
# #     except Exception as e:
# #         st.warning(f"⚠️ Model could not be loaded: {e}")

# # # 📌 **Home Page**
# # if page == "🏠 Home":
# #     st.title("Rain in Australia 🌧️🌦️")
# #     st.image(r"C:\Users\NohaA\myenv\finallllllll project\New folder\Screenshot 2025-03-09 201614.png", width=600)
    
# #     if df is not None:
# #         st.subheader("Dataset Preview")
# #         st.dataframe(df.head())
# #     else:
# #         st.error("🚨 Dataset could not be loaded.")
    
# #     st.write("Use the sidebar to navigate to Visualization or Prediction.")

# # # 📌 **Visualization Page**
# # elif page == "📊 Visualization" and df is not None:
# #     st.title("📊 Weather Data Visualization")
# #     st.sidebar.title("🔍 Choose Visualization")
# #     visualization_option = st.sidebar.selectbox("Select an option", [
# #         "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
# #         "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
# #     ])
    
# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     if visualization_option == "🌧️ Rainfall Distribution":
# #         selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
# #         sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
# #         ax.set_title(f"Rainfall Distribution in {selected_location}")

# #     elif visualization_option == "📊 Rain Probability by Wind Direction":
# #         df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
# #         sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
# #         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# #     elif visualization_option == "🔥 Correlation Heatmap":
# #         sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

# #     elif visualization_option == "💨 Wind Speed vs. Rain Probability":
# #         sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
# #         ax.set_title("Wind Speed in Morning vs. Rain Probability")

# #     elif visualization_option == "🌦️ Seasonal Rain Effects":
# #         sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
# #         ax.set_title("Seasonal Effect on Rain Tomorrow")
    
# #     st.pyplot(fig)

# # # 📌 **Prediction Page**
# # elif page == "🔍 Prediction" and df is not None:
# #     st.title("🔍 Rain Prediction")
    
# #     st.sidebar.title("🌍 Enter Weather Details")
# #     location = st.sidebar.selectbox("🌏 Location", df["Location"].unique())
# #     sunshine = st.sidebar.slider("☀️ Sunshine (hours)", 0.0, 15.0, 5.0, step=0.5)
# #     wind_gust_dir = st.sidebar.selectbox("💨 Wind Gust Direction", df["WindGustDir"].unique())
# #     wind_gust_speed = st.sidebar.slider("🌬️ Wind Gust Speed (km/h)", 0, 100, 30)
# #     humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
# #     humidity_3pm = st.sidebar.slider("💧 Humidity 3PM (%)", 0, 100, 50)
# #     prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
# #     rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

# #     # Prepare input features
# #     input_features = np.array([[hash(location) % 1000, sunshine, hash(wind_gust_dir) % 100, wind_gust_speed,
# #                                 humidity_9am, humidity_3pm, prev_day_rainfall, int(rain_today)]])

# #     if model is None:
# #         st.warning("⚠️ Model is not available. Prediction feature is disabled.")
# #     else:
# #         if st.sidebar.button("Submit"):
# #             prediction = model.predict(input_features)
# #             st.success("☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain.")
# ##########################################################################################


# ###########################################################################################



# ## correct 



# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import os
# # import joblib

# # # ✅ Set page config
# # st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# # # Sidebar Navigation
# # st.sidebar.title("Navigation")
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# # # ✅ Define the missing function before loading the model
# # def custom_function(x):
# #     return x  # Replace this with actual function logic if needed

# # # ✅ Load the dataset
# # @st.cache_data
# # def load_data():
# #     file_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\weatherAUS.csv"
# #     if not os.path.exists(file_path):
# #         st.error("❌ Dataset file not found! Check the file path.")
# #         return None
# #     df = pd.read_csv(file_path, parse_dates=["Date"])
# #     df.dropna(subset=["WindGustDir", "RainTomorrow", "Location", "Rainfall", 
# #                       "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"], inplace=True)
# #     df["Season"] = df["Date"].dt.month.map(
# #         lambda m: "Winter" if m in [12, 1, 2] else 
# #                   "Spring" if m in [3, 4, 5] else 
# #                   "Summer" if m in [6, 7, 8] else 
# #                   "Autumn"
# #     )
# #     return df

# # df = load_data()

# # # ✅ Load Model
# # model_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting3_pipeline.pkl"
# # model = None

# # if os.path.exists(model_path):
# #     try:
# #         model = joblib.load(model_path)
# #         st.sidebar.success("✅ Model Loaded Successfully!")
# #     except AttributeError as e:
# #         st.sidebar.error(f"❌ Model error: {e}. Ensure all dependencies are defined.")
# #     except Exception as e:
# #         st.sidebar.error(f"❌ General error loading model: {e}")
# # else:
# #     st.sidebar.warning("⚠️ Model file not found.")

# # # 📌 **Home Page**
# # if page == "🏠 Home":
# #     st.title("Rain in Australia 🌧️🌦️")
# #     st.image(r"C:\Users\NohaA\myenv\finallllllll project\New folder\Screenshot 2025-03-09 201614.png", width=600)
    
# #     if df is not None:
# #         st.subheader("Dataset Preview")
# #         st.dataframe(df.head())
# #     else:
# #         st.error("🚨 Dataset could not be loaded.")
    
# #     st.write("Use the sidebar to navigate to Visualization or Prediction.")

# # # 📌 **Visualization Page**
# # elif page == "📊 Visualization" and df is not None:
# #     st.title("📊 Weather Data Visualization")
# #     st.sidebar.title("🔍 Choose Visualization")
# #     visualization_option = st.sidebar.selectbox("Select an option", [
# #         "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
# #         "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
# #     ])
    
# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     if visualization_option == "🌧️ Rainfall Distribution":
# #         selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
# #         sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
# #         ax.set_title(f"Rainfall Distribution in {selected_location}")

# #     elif visualization_option == "📊 Rain Probability by Wind Direction":
# #         df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
# #         sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
# #         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# #     elif visualization_option == "🔥 Correlation Heatmap":
# #         sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

# #     elif visualization_option == "💨 Wind Speed vs. Rain Probability":
# #         sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
# #         ax.set_title("Wind Speed in Morning vs. Rain Probability")

# #     elif visualization_option == "🌦️ Seasonal Rain Effects":
# #         sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
# #         ax.set_title("Seasonal Effect on Rain Tomorrow")
    
# #     st.pyplot(fig)

# # # 📌 **Prediction Page**
# # elif page == "🔍 Prediction" and df is not None:
# #     st.title("🔍 Rain Prediction")
    
# #     st.sidebar.title("🌍 Enter Weather Details")
# #     location = st.sidebar.selectbox("🌏 Location", df["Location"].unique())
# #     sunshine = st.sidebar.slider("☀️ Sunshine (hours)", 0.0, 15.0, 5.0, step=0.5)
# #     wind_gust_dir = st.sidebar.selectbox("💨 Wind Gust Direction", df["WindGustDir"].unique())
# #     wind_gust_speed = st.sidebar.slider("🌬️ Wind Gust Speed (km/h)", 0, 100, 30)
# #     humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
# #     humidity_3pm = st.sidebar.slider("💧 Humidity 3PM (%)", 0, 100, 50)
# #     prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
# #     rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

# #     # Prepare input features
# #     input_df = pd.DataFrame({
# #         "Location": [location],
# #         "Sunshine": [sunshine],
# #         "WindGustDir": [wind_gust_dir],
# #         "WindGustSpeed": [wind_gust_speed],
# #         "Humidity9am": [humidity_9am],
# #         "Humidity3pm": [humidity_3pm],
# #         "Rainfall": [prev_day_rainfall],
# #         "RainToday": [int(rain_today)]
# #     })

# #     # One-hot encoding for categorical variables
# #     input_df = pd.get_dummies(input_df, columns=["Location", "WindGustDir"])

# #     if model is None:
# #         st.warning("⚠️ Model is not available. Prediction feature is disabled.")
# #     else:
# #         if st.sidebar.button("Submit"):
# #             try:
# #                 prediction = model.predict(input_df)
# #                 result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
# #                 st.success(result)
# #             except Exception as e:
# #                 st.error(f"⚠️ Prediction error: {e}")

# #########################################################################
# # import streamlit as st
# # import pandas as pd
# # import joblib
# # import os

# # # ✅ Set page config
# # st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# # # Sidebar Navigation
# # st.sidebar.title("Navigation")
# # page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# # # ✅ Load Model
# # model_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting3_pipeline.pkl"
# # model = None

# # if os.path.exists(model_path):
# #     try:
# #         model = joblib.load(model_path)
# #         if not hasattr(model, "predict"):  
# #             st.sidebar.error("❌ Model is not properly trained. Retrain and save it again.")
# #             model = None  
# #         else:
# #             st.sidebar.success("✅ Model Loaded Successfully!")
# #     except Exception as e:
# #         st.sidebar.error(f"❌ Error loading model: {e}")
# #         model = None
# # else:
# #     st.sidebar.warning("⚠️ Model file not found.")

# # # ✅ **Prediction Page**
# # if page == "🔍 Prediction":
# #     st.title("🔍 Rain Prediction")

# #     # Sample locations (Adjust based on your dataset)
# #     locations = ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth"]
# #     wind_directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

# #     # User input
# #     location = st.selectbox("🌏 Location", locations)
# #     sunshine = st.slider("☀️ Sunshine (hours)", 0.0, 15.0, 5.0, step=0.5)
# #     wind_gust_dir = st.selectbox("💨 Wind Gust Direction", wind_directions)
# #     wind_gust_speed = st.slider("🌬️ Wind Gust Speed (km/h)", 0, 100, 30)
# #     humidity_9am = st.slider("💧 Humidity 9AM (%)", 0, 100, 50)
# #     humidity_3pm = st.slider("💧 Humidity 3PM (%)", 0, 100, 50)
# #     prev_day_rainfall = st.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
# #     rain_today = 1 if st.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes" else 0

# #     # ✅ Prepare input dataframe
# #     input_data = pd.DataFrame({
# #         "Location": [location], "Sunshine": [sunshine], "WindGustDir": [wind_gust_dir],
# #         "WindGustSpeed": [wind_gust_speed], "Humidity9am": [humidity_9am],
# #         "Humidity3pm": [humidity_3pm], "Rainfall": [prev_day_rainfall], "RainToday": [rain_today]
# #     })

# #     # Convert categorical columns
# #     input_data = pd.get_dummies(input_data, columns=["Location", "WindGustDir"])

# #     # Ensure correct feature alignment
# #     if model:
# #         try:
# #             # Get model feature names (assuming pipeline includes preprocessing)
# #             expected_features = model.feature_names_in_
# #             input_data = input_data.reindex(columns=expected_features, fill_value=0)

# #             if st.button("Submit"):
# #                 prediction = model.predict(input_data)
# #                 result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
# #                 st.success(result)

# #         except Exception as e:
# #             st.error(f"⚠️ Prediction error: {e}")

# #     else:
# #         st.warning("⚠️ Model is not available. Prediction feature is disabled.")
# ###################################################################################







# ########################################


# # #####################################################
# ######################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# import joblib

# # ✅ Set Streamlit Page
# st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# # ✅ Sidebar Navigation
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# # ✅ Load the dataset
# @st.cache_data
# def load_data():
#     file_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\weatherAUS.csv"
#     if not os.path.exists(file_path):
#         st.error("❌ Dataset file not found! Check the file path.")
#         return None
#     df = pd.read_csv(file_path, parse_dates=["Date"])
    
#     # Drop missing values for key columns
#     df.dropna(subset=["WindGustDir", "RainTomorrow", "Location", "Rainfall", 
#                       "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"], inplace=True)
    
#     # Add season column
#     df["Season"] = df["Date"].dt.month.map(
#         lambda m: "Winter" if m in [12, 1, 2] else 
#                   "Spring" if m in [3, 4, 5] else 
#                   "Summer" if m in [6, 7, 8] else 
#                   "Autumn"
#     )
#     return df

# df = load_data()

# # ✅ Load Model
# model_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting5_pipeline.pkl"
# model = None

# if os.path.exists(model_path):
#     try:
#         model = joblib.load(model_path)
#         if hasattr(model, "predict"):
#             st.sidebar.success("✅ Model Loaded Successfully!")
#         else:
#             st.sidebar.error("❌ Model is not properly trained. Retrain and save it again.")
#             model = None  # Prevent using an untrained model
#     except AttributeError as e:
#         st.sidebar.error(f"❌ Model error: {e}. Ensure all dependencies are defined.")
#     except Exception as e:
#         st.sidebar.error(f"❌ General error loading model: {e}")
# else:
#     st.sidebar.warning("⚠️ Model file not found.")

# # 📌 **Home Page**
# if page == "🏠 Home":
#     st.title("Rain in Australia 🌧️🌦️")
#     st.image(r"C:\Users\NohaA\myenv\finallllllll project\New folder\Screenshot 2025-03-09 201614.png", width=600)
    
#     if df is not None:
#         st.subheader("Dataset Preview")
#         st.dataframe(df.head())
#     else:
#         st.error("🚨 Dataset could not be loaded.")
    
#     st.write("Use the sidebar to navigate to Visualization or Prediction.")

# # 📌 **Visualization Page**
# elif page == "📊 Visualization" and df is not None:
#     st.title("📊 Weather Data Visualization")
#     st.sidebar.title("🔍 Choose Visualization")
#     visualization_option = st.sidebar.selectbox("Select an option", [
#         "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
#         "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
#     ])
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     if visualization_option == "🌧️ Rainfall Distribution":
#         selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
#         sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
#         ax.set_title(f"Rainfall Distribution in {selected_location}")

#     elif visualization_option == "📊 Rain Probability by Wind Direction":
#         df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
#         sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

#     elif visualization_option == "🔥 Correlation Heatmap":
#         sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

#     elif visualization_option == "💨 Wind Speed vs. Rain Probability":
#         sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
#         ax.set_title("Wind Speed in Morning vs. Rain Probability")

#     elif visualization_option == "🌦️ Seasonal Rain Effects":
#         sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
#         ax.set_title("Seasonal Effect on Rain Tomorrow")
    
#     st.pyplot(fig)

# # 📌 **Prediction Page**
# elif page == "🔍 Prediction" and df is not None:
#     st.title("🔍 Rain Prediction")
    
#     st.sidebar.title("🌍 Enter Weather Details")
#     location = st.sidebar.selectbox("🌏 Location", df["Location"].unique())
#     sunshine = st.sidebar.slider("☀️ Sunshine (hours)", 0.0, 15.0, 5.0, step=0.5)
#     wind_gust_dir = st.sidebar.selectbox("💨 Wind Gust Direction", df["WindGustDir"].unique())
#     wind_gust_speed = st.sidebar.slider("🌬️ Wind Gust Speed (km/h)", 0, 100, 30)
#     humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
#     humidity_3pm = st.sidebar.slider("💧 Humidity 3PM (%)", 0, 100, 50)
#     prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
#     rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

#     # Prepare input features
#     input_df = pd.DataFrame({
#         "Location": [location],
#         "Sunshine": [sunshine],
#         "WindGustDir": [wind_gust_dir],
#         "WindGustSpeed": [wind_gust_speed],
#         "Humidity9am": [humidity_9am],
#         "Humidity3pm": [humidity_3pm],
#         "Rainfall": [prev_day_rainfall],
#         "RainToday": [int(rain_today)]
#     })

#     # One-hot encoding for categorical variables
#     input_df = pd.get_dummies(input_df, columns=["Location", "WindGustDir"])

#     if model is None:
#         st.warning("⚠️ Model is not available. Prediction feature is disabled.")
#     else:
#         if st.sidebar.button("Submit"):
#             try:
#                 prediction = model.predict(input_df)
#                 result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
#                 st.success(result)
#             except Exception as e:
#                 st.error(f"⚠️ Prediction error: {e}")
##################################################################################
###################################################################################

## finall correct 


# import streamlit as st
# import joblib
# import pandas as pd
# import os
# import numpy as np

# # ✅ Set Streamlit Page
# st.set_page_config(page_title="Rain Prediction", page_icon="☔", layout="wide")

# # ✅ Define Model Paths (Use first available)
# model_paths = [
#     r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting5_pipeline.pkl", 
#     "voting3_pipeline.pkl"
# ]

# model = None
# scaler = None
# le = None

# # ✅ Load Model
# for path in model_paths:
#     if os.path.exists(path):
#         try:
#             model, scaler, le = joblib.load(path)
#             if hasattr(model, "predict"):
#                 st.sidebar.success(f"✅ Model Loaded Successfully from {path}!")
#             else:
#                 st.sidebar.error("❌ Model is not properly trained. Retrain and save it again.")
#                 model = None
#             break
#         except Exception as e:
#             st.sidebar.error(f"❌ Error loading model from {path}: {e}")
#             model = None

# if model is None:
#     st.sidebar.warning("⚠️ No valid model file found. Please retrain the model.")

# # ✅ **Prediction Page**
# st.title("🔍 Rain Prediction")

# if model is None:
#     st.warning("⚠️ Model is not available. Prediction feature is disabled.")
# else:
#     st.success("🎯 Model is ready for predictions!")

#     # Sidebar Input Fields
#     st.sidebar.title("🌍 Enter Weather Details")
#     location = st.sidebar.text_input("🌏 Location (Type the city name)")
#     wind_gust_dir = st.sidebar.text_input("💨 Wind Gust Direction (N, S, E, W, etc.)")
#     wind_speed_9am = st.sidebar.slider("🌬️ Wind Speed 9AM (km/h)", 0, 100, 30)
#     humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
#     temp_9am = st.sidebar.slider("🌡️ Temperature 9AM (°C)", -10, 50, 20)
#     temp_3pm = st.sidebar.slider("🌡️ Temperature 3PM (°C)", -10, 50, 25)
#     prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
#     rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

#     # ✅ Convert categorical inputs
#     try:
#         location_encoded = le.transform([location])[0] if location else 0
#         wind_gust_encoded = le.transform([wind_gust_dir])[0] if wind_gust_dir else 0
#     except ValueError:
#         st.error("⚠️ Invalid Location or Wind Gust Direction! Check the spelling or dataset.")
#         location_encoded, wind_gust_encoded = 0, 0

#     # ✅ Prepare input features
#     input_data = np.array([
#         location_encoded, wind_gust_encoded, wind_speed_9am, humidity_9am, temp_9am, temp_3pm, prev_day_rainfall, int(rain_today)
#     ]).reshape(1, -1)

#     # ✅ Scale input
#     input_data_scaled = scaler.transform(input_data)

#     # ✅ Predict button
#     if st.sidebar.button("Submit"):
#         try:
#             prediction = model.predict(input_data_scaled)
#             result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
#             st.success(result)
#         except Exception as e:
#             st.error(f"⚠️ Prediction error: {e}")

##$$$$$$$$$$$$$$$###################$$$$$$$$$$###################################

#########################################
import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Set Streamlit Page
st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# ✅ Load dataset
@st.cache_data
def load_data():
    file_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\weatherAUS.csv"
    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found! Check the file path.")
        return None
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.dropna(subset=["WindGustDir", "RainTomorrow", "Location", "Rainfall", "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"], inplace=True)
    df["Season"] = df["Date"].dt.month.map(
        lambda m: "Winter" if m in [12, 1, 2] else 
                  "Spring" if m in [3, 4, 5] else 
                  "Summer" if m in [6, 7, 8] else 
                  "Autumn"
    )
    return df

df = load_data()

# ✅ Load Model
model_paths = [
    r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting5_pipeline.pkl", 
    "voting3_pipeline.pkl"
]

model, scaler, le = None, None, None
for path in model_paths:
    if os.path.exists(path):
        try:
            model, scaler, le = joblib.load(path)
            if hasattr(model, "predict"):
                st.sidebar.success(f"✅ Model Loaded Successfully from {path}!")
            else:
                st.sidebar.error("❌ Model is not properly trained. Retrain and save it again.")
                model = None
            break
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model from {path}: {e}")
            model = None

if model is None:
    st.sidebar.warning("⚠️ No valid model file found. Please retrain the model.")

# 📌 **Home Page**
if page == "🏠 Home":
    st.title("Rain in Australia 🌧️🌦️")
    st.image(r"C:\Users\NohaA\myenv\finallllllll project\New folder\Screenshot 2025-03-09 201614.png", width=600)
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    else:
        st.error("🚨 Dataset could not be loaded.")
    
    st.write("Use the sidebar to navigate to Visualization or Prediction.")

# 📌 **Visualization Page**
elif page == "📊 Visualization" and df is not None:
    st.title("📊 Weather Data Visualization")
    st.sidebar.title("🔍 Choose Visualization")
    visualization_option = st.sidebar.selectbox("Select an option", [
        "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
        "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
    ])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if visualization_option == "🌧️ Rainfall Distribution":
        selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
        sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
        ax.set_title(f"Rainfall Distribution in {selected_location}")

    elif visualization_option == "📊 Rain Probability by Wind Direction":
        df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    elif visualization_option == "🔥 Correlation Heatmap":
        sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

    elif visualization_option == "💨 Wind Speed vs. Rain Probability":
        sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
        ax.set_title("Wind Speed in Morning vs. Rain Probability")

    elif visualization_option == "🌦️ Seasonal Rain Effects":
        sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
        ax.set_title("Seasonal Effect on Rain Tomorrow")
    
    st.pyplot(fig)

# 📌 **Prediction Page**
elif page == "🔍 Prediction" and model is not None:
    st.title("🔍 Rain Prediction")
    st.sidebar.title("🌍 Enter Weather Details")
    location = st.sidebar.text_input("🌏 Location (Type the city name)")
    wind_gust_dir = st.sidebar.text_input("💨 Wind Gust Direction (N, S, E, W, etc.)")
    wind_speed_9am = st.sidebar.slider("🌬️ Wind Speed 9AM (km/h)", 0, 100, 30)
    humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
    temp_9am = st.sidebar.slider("🌡️ Temperature 9AM (°C)", -10, 50, 20)
    temp_3pm = st.sidebar.slider("🌡️ Temperature 3PM (°C)", -10, 50, 25)
    prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
    rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

    try:
        location_encoded = le.transform([location])[0] if location else 0
        wind_gust_encoded = le.transform([wind_gust_dir])[0] if wind_gust_dir else 0
    except ValueError:
        st.error("⚠️ Invalid Location or Wind Gust Direction! Check the spelling or dataset.")
        location_encoded, wind_gust_encoded = 0, 0

    input_data = np.array([
        location_encoded, wind_gust_encoded, wind_speed_9am, humidity_9am, temp_9am, temp_3pm, prev_day_rainfall, int(rain_today)
    ]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    if st.sidebar.button("Submit"):
        prediction = model.predict(input_data_scaled)
        result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
        st.success(result)
