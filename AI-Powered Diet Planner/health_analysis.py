import matplotlib.pyplot as plt
from io import BytesIO
import base64

def analyze_nutrition(nutrition_dict, goal="General"):
    # Extracting nutrition values from the nutrition_dict
    protein = nutrition_dict.get("protein", 0)
    carbs = nutrition_dict.get("carbs", 0)
    fat = nutrition_dict.get("fat", 0)
    calories = nutrition_dict.get("calories", 0)

    # Default health verdict
    verdict = "Looks good for a general diet."

    # Weight Loss Goal Analysis
    if goal == "Weight Loss":
        if calories > 600:
            verdict = "⚠️ Too high in calories for a weight loss diet."
        elif fat > 20:
            verdict = "⚠️ Contains too much fat for a weight loss goal."
        else:
            verdict = "✅ Great fit for weight loss!"

    # Muscle Gain Goal Analysis
    elif goal == "Muscle Gain":
        if protein < 20:
            verdict = "⚠️ Needs more protein for muscle gain."
        else:
            verdict = "✅ Solid protein content for muscle growth!"

    # Low Sodium Diet Analysis (If Sodium is available)
    elif goal == "Low Sodium Diet":
        sodium = nutrition_dict.get("sodium", 0)
        if sodium > 1000:
            verdict = "⚠️ Too much sodium for a low-sodium diet."
        else:
            verdict = "✅ Great for a low-sodium diet!"

    # General Analysis (if no specific goal is selected)
    else:
        if calories > 700:
            verdict = "⚠️ High in calories for general health."
        if fat > 30:
            verdict = "⚠️ High in fat for general health."
        elif protein < 10:
            verdict = "⚠️ Low in protein, which may not be ideal for a balanced diet."

    # Nutritional Breakdown
    analysis = f"""
    **Calories:** {calories} kcal  
    **Protein:** {protein} g  
    **Carbs:** {carbs} g  
    **Fat:** {fat} g  
    """

    # Pie chart visualization
    labels = ['Protein', 'Carbs', 'Fat']
    values = [protein, carbs, fat]
    colors = ['#ff9999','#66b3ff','#99ff99']

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert the plot to a base64 string to display in Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.getvalue()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    buf.close()

    return verdict, analysis, base64_img


# # health_analysis.py
# import matplotlib.pyplot as plt
# import io
# import base64
# import re
# import numpy as np

# def extract_nutrient(nutrition_text, nutrient_name):
#     pattern = rf"{nutrient_name}\s+([\d.]+)"
#     match = re.search(pattern, nutrition_text)
#     if match:
#         return float(match.group(1))
#     return 0.0  # Default to 0 if not found

# def analyze_nutrition(nutrition_text, goal="General"):
#     # Extract relevant nutrients
#     calories = extract_nutrient(nutrition_text, "Calories")
#     protein = extract_nutrient(nutrition_text, "Protein")
#     carbs = extract_nutrient(nutrition_text, "Carbohydrates")
#     fat = extract_nutrient(nutrition_text, "Fat")
#     sodium = extract_nutrient(nutrition_text, "Sodium")

#     analysis = []
#     verdict = ""

#     if goal == "Weight Loss":
#         if calories > 500:
#             analysis.append("⚠️ High in calories for a weight loss diet.")
#         else:
#             analysis.append("✅ Calorie content is suitable for weight loss.")
#         if fat > 20:
#             analysis.append("⚠️ Too much fat for weight management.")
#     elif goal == "Muscle Gain":
#         if protein >= 25:
#             analysis.append("💪 Good protein content for muscle gain.")
#         else:
#             analysis.append("⚠️ Not enough protein for muscle building.")
#         if calories < 400:
#             analysis.append("⚠️ Might be low on calories for bulking.")
#     elif goal == "Low Sodium Diet":
#         if sodium > 800:
#             analysis.append("⚠️ High in sodium, not ideal for low-sodium diets.")
#         else:
#             analysis.append("✅ Sodium levels are within healthy range.")
#     else:
#         analysis.append("ℹ️ General health overview provided.")

#     verdict = "Looks decent overall! 🥗" if not any("⚠️" in line for line in analysis) else "Not the healthiest pick 😬"

#     return verdict, "\n".join(analysis), generate_pie_chart(carbs, fat, protein)

# def generate_pie_chart(carbs, fat, protein):
#     labels = ['Carbohydrates', 'Fat', 'Protein']
#     values = [carbs, fat, protein]
    
#     # Avoid dividing by zero
#     if sum(values) == 0:
#         values = [1, 1, 1]

#     colors = ['#FF9999', '#FFCC99', '#99CCFF']
#     fig, ax = plt.subplots()
#     ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
#     ax.axis('equal')

#     # Save to buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)
#     img_base64 = base64.b64encode(buf.read()).decode("utf-8")
#     return img_base64
