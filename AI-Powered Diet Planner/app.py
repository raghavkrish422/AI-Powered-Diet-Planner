import streamlit as st
from PIL import Image
from food_classifier import classify_image
from nutrition import get_nutrition
from health_analysis import analyze_nutrition
import google.generativeai as genai

# Hardcode the Gemini API key
api_key = "AIzaSyCHGtzAxtXSPw6vaVdSshsag7lvocPTQVA"  # Replace with your actual Gemini API key

# Initialize the Gemini model with the hardcoded API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

st.set_page_config(page_title="Snap & Snack - AI Diet Planner", layout="centered")
st.title("🍽️ Snap & Snack - AI Diet Planner")

# Upload Image
uploaded_file = st.file_uploader("📷 Upload your food image", type=["jpg", "jpeg", "png"])

# Gemini Setup
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return [{"mime_type": uploaded_file.type, "data": bytes_data}]
    else:
        return None

def get_gemini_response(prompt, image_parts):
    response = model.generate_content([prompt, image_parts[0]])
    return response.text

# Prompt for Gemini food analysis
input_prompt = """
You are an expert nutritionist analyzing the food items in the image.
Start by determining if the image contains food items. If not, say "No food items detected in the image."

If food is present:
- Name the meal
- List ingredients with estimated calories
- Give total calories
- Mention if it is healthy or not
- Estimate % split of protein, carbs, fats
- Mention total fiber content

Example format:
Meal Name: ...
1. Ingredient - calories
...
Total estimated calories: ...
Health Verdict: ...
Protein: X%, Carbs: Y%, Fats: Z%
Fiber: ...g
"""

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analyzing Image..."):
            food_item = classify_image(image)

        if not food_item:
            st.error("⚠️ Could not classify the food. Please try another image.")
        else:
            food_item = food_item.strip()
            st.success(f"🍔 Detected Food: **{food_item}**")

            st.info("📦 Fetching Nutrition Info...")
            result, nutrition_dict = get_nutrition(food_item)

            if result.startswith("❌"):
                st.error(result)
            else:
                # Nutritional Info
                st.subheader("🍏 Nutritional Info")
                st.markdown(result)

                # Goal
                goal = st.selectbox("🎯 Choose your diet goal", ["General", "Weight Loss", "Muscle Gain", "Low Sodium Diet"])

                # Analyze
                verdict, analysis, chart = analyze_nutrition(nutrition_dict, goal)
                st.subheader("📌 Health Verdict")
                st.success(verdict)
                st.markdown(analysis)

                st.subheader("📊 Macronutrient Distribution")
                st.image(f"data:image/png;base64,{chart}", use_column_width=True)

                st.subheader("🧠 AI Expert Analysis (NutritionGPT)")
                with st.spinner("NutritionGPT is analyzing the food..."):
                    image_data = input_image_setup(uploaded_file)
                    gemini_analysis = get_gemini_response(input_prompt, image_data)
                    st.success("NutritionGPT Analysis Complete!")
                    st.text_area("NutritionGPT Food Report", gemini_analysis, height=300)

                st.subheader("💬 Ask nutritionGpt About This Food")
                user_question = st.text_input("Ask any question about the food in the image")
                if user_question:
                    with st.spinner("NutritionGPT is thinking..."):
                        follow_up = get_gemini_response(user_question, image_data)
                        st.write(follow_up)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")


## Final--working--code-----------
# import streamlit as st
# from PIL import Image
# from food_classifier import classify_image
# from nutrition import get_nutrition
# from health_analysis import analyze_nutrition

# st.set_page_config(page_title="Snap & Snack - AI Diet Planner", layout="centered")
# st.title("🍽️ Snap & Snack - AI Diet Planner")

# ## Google-credentials : 281530409186-srnpdugjc3ksvlm9sm8isv4fibrpgc7s.apps.googleusercontent.com
# # Step 1: Upload Image
# uploaded_file = st.file_uploader("📷 Upload your food image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     try:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="📸 Uploaded Image", use_column_width=True)

#         with st.spinner("🔍 Analyzing Image..."):
#             food_item = classify_image(image)

#         if not food_item:
#             st.error("⚠️ Could not classify the food. Please try another image.")
#         else:
#             food_item = food_item.strip()
#             st.success(f"🍔 Detected Food: **{food_item}**")

#             st.info("📦 Fetching Nutrition Info...")
#             result, nutrition_dict = get_nutrition(food_item)

#             if result.startswith("❌"):
#                 st.error(result)
#             else:
#                 # Step 2: Show Nutritional Info
#                 st.subheader("🍏 Nutritional Info")
#                 st.markdown(result)

#                 # Step 3: Choose Goal
#                 goal = st.selectbox("🎯 Choose your diet goal", ["General", "Weight Loss", "Muscle Gain", "Low Sodium Diet"])

#                 # Step 4: Analyze Nutrition
#                 verdict, analysis, chart = analyze_nutrition(nutrition_dict, goal)

#                 st.subheader("📌 Health Verdict")
#                 st.success(verdict)
#                 st.markdown(analysis)

#                 # Step 5: Show Macronutrient Chart
#                 st.subheader("📊 Macronutrient Distribution")
#                 st.image(f"data:image/png;base64,{chart}", use_column_width=True)

#     except Exception as e:
#         st.error(f"❌ An error occurred: {e}")



# import streamlit as st
# from PIL import Image
# from food_classifier import classify_image
# from nutrition import get_nutrition
# from health_analysis import analyze_nutrition

# st.title("🍽️ Snap & Snack - AI Diet Planner")

# # Step 1: Upload Image
# uploaded_file = st.file_uploader("Upload your food image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     with st.spinner("🔍 Analyzing Image..."):
#         food_item = classify_image(image)  # Convert image to food name
#         st.success(f"🍔 Detected Food: **{food_item}**")

#         st.info("📦 Fetching Nutrition Info...")
#         result = get_nutrition(food_item)  # Get nutritional data

#         if result.startswith("❌"):
#             st.error(result)
#         else:
#             # Step 2: Show Nutritional Info
#             st.subheader("🍏 Nutritional Info")
#             st.markdown(result)

#             # Step 3: Choose Goal
#             goal = st.selectbox("🎯 Choose your diet goal", ["General", "Weight Loss", "Muscle Gain", "Low Sodium Diet"])

#             # Step 4: Analyze Nutrition
#             verdict, analysis, chart = analyze_nutrition(result, goal)

#             st.subheader("📌 Health Verdict")
#             st.success(verdict)
#             st.markdown(analysis)

#             # Step 5: Show Macronutrient Chart
#             st.subheader("📊 Macronutrient Distribution")
#             st.image(f"data:image/png;base64,{chart}", use_column_width=True)

#############################################################################
# import streamlit as st
# from PIL import Image
# from food_classifier import classify_image
# from nutrition import get_nutrition
# from health_analysis import analyze_nutrition

# st.title("🍽️ Snap & Snack - AI Diet Planner")

# uploaded_file = st.file_uploader("Upload your food image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     with st.spinner("Analyzing Image..."):
#         food_item = classify_image(image) ## Converting image to food_item
#         st.success(f"Detected Food: **{food_item}**")

#         st.info("Fetching Nutrition Info...")
#         result = get_nutrition(food_item) ### Tells the nutrients in the food

#         if result.startswith("❌"):
#             st.error(result)
#         else:
#             st.subheader("🍏 Nutritional Info")
#             st.markdown(result)
# # Optional: let user select their diet goal
# goal = st.selectbox("Choose your diet goal", ["General", "Weight Loss", "Muscle Gain", "Low Sodium Diet"])

# # After showing the markdown result
# verdict, analysis, chart = analyze_nutrition(result, goal)

# st.subheader("📌 Health Verdict")
# st.success(verdict)
# st.markdown(analysis)

# # Show pie chart
# st.subheader("📊 Macronutrient Distribution")
# st.image(f"data:image/png;base64,{chart}", use_column_width=True)

############################------WC------##########
# import streamlit as st
# from PIL import Image
# from food_classifier import classify_image
# from nutrition import get_nutrition

# st.title("🍽️ Snap & Snack - AI Diet Planner")

# uploaded_file = st.file_uploader("Upload your food image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     with st.spinner("Analyzing Image..."):
#         food_item = classify_image(image)
#         st.success(f"Detected Food: **{food_item}**")

#         st.info("Fetching Nutrition Info...")
#         data = get_nutrition(food_item)
        
#         if 'error' in data:
#             st.error(data['error'])
#         else:
#             st.subheader("🍏 Nutritional Info")
#             st.write(f"**Calories:** {data.get('calories')} kcal")
#             st.write(f"**Protein:** {data.get('protein')} g")
#             st.write(f"**Carbs:** {data.get('carbs')} g")
#             st.write(f"**Fats:** {data.get('fat')} g")