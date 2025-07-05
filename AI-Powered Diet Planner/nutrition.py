##### Working-03
import requests

APP_ID = "55b6e996"
APP_KEY = "fa595b959fd39e454fa4627608bf20ab"

def get_nutrition(food_item):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": APP_ID,
        "x-app-key": APP_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "query": food_item
    }   

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        if "foods" not in data or len(data["foods"]) == 0:
            return "❌ No nutritional data found for the given item.", {}

        food = data["foods"][0]

        # Table with actual values and expected range
        table = [
            ("Calories", f"{round(food.get('nf_calories', 0))} kcal (Expected: 450–550 kcal)"),
            ("Protein", f"{food.get('nf_protein', 0)} g (Expected: 20–25 g)"),
            ("Carbohydrates", f"{food.get('nf_total_carbohydrate', 0)} g (Expected: 30–35 g)"),
            ("Fat", f"{food.get('nf_total_fat', 0)} g (Expected: 25–35 g)"),
            ("Saturated Fat", f"{food.get('nf_saturated_fat', 0)} g (Expected: 10–15 g)"),
            ("Cholesterol", f"{food.get('nf_cholesterol', 0)} mg (Expected: 60–80 mg)"),
            ("Fiber", f"{food.get('nf_dietary_fiber', 0)} g (Expected: 2–3 g)"),
            ("Sugars", f"{food.get('nf_sugars', 0)} g (Expected: 5–7 g)"),
            ("Sodium", f"{food.get('nf_sodium', 0)} mg (Expected: 800–1200 mg)"),
            ("Calcium", "150–200 mg (estimated, not from API)"),
            ("Iron", "2–3 mg (estimated, not from API)")
        ]

        table_str = f"📊 Nutritional Info for **{food.get('food_name', 'Unknown')}**:\n\n"
        table_str += "| Nutrient       | Value |\n"
        table_str += "|----------------|-------------------------------|\n"
        for name, value in table:
            table_str += f"| {name:<14} | {value} |\n"

        nutrition_dict = {
            "calories": round(food.get('nf_calories', 0)),
            "protein": int(food.get('nf_protein', 0)),
            "carbs": int(food.get('nf_total_carbohydrate', 0)),
            "fat": int(food.get('nf_total_fat', 0)),
        }

        return table_str, nutrition_dict

    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {str(e)}", {}
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", {}


#####-Working-code-02########
# import requests

# APP_ID = "55b6e996"
# APP_KEY = "fa595b959fd39e454fa4627608bf20ab"

# def get_nutrition(food_item):
#     url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
#     headers = {
#         "x-app-id": APP_ID,
#         "x-app-key": APP_KEY,
#         "Content-Type": "application/json"
#     }
#     body = {
#         "query": food_item
#     }

#     try:
#         response = requests.post(url, headers=headers, json=body)
#         response.raise_for_status()
#         data = response.json()

#         if "foods" not in data or len(data["foods"]) == 0:
#             return "❌ No nutritional data found for the given item."

#         food = data["foods"][0]

#         # Table with actual values and expected range
#         table = [
#             ("Calories", f"{round(food.get('nf_calories', 0))} kcal (Expected: 450–550 kcal)"),
#             ("Protein", f"{food.get('nf_protein', 0)} g (Expected: 20–25 g)"),
#             ("Carbohydrates", f"{food.get('nf_total_carbohydrate', 0)} g (Expected: 30–35 g)"),
#             ("Fat", f"{food.get('nf_total_fat', 0)} g (Expected: 25–35 g)"),
#             ("Saturated Fat", f"{food.get('nf_saturated_fat', 0)} g (Expected: 10–15 g)"),
#             ("Cholesterol", f"{food.get('nf_cholesterol', 0)} mg (Expected: 60–80 mg)"),
#             ("Fiber", f"{food.get('nf_dietary_fiber', 0)} g (Expected: 2–3 g)"),
#             ("Sugars", f"{food.get('nf_sugars', 0)} g (Expected: 5–7 g)"),
#             ("Sodium", f"{food.get('nf_sodium', 0)} mg (Expected: 800–1200 mg)"),
#             ("Calcium", "150–200 mg (estimated, not from API)"),
#             ("Iron", "2–3 mg (estimated, not from API)")
#         ]

#         table_str = f"📊 Nutritional Info for **{food.get('food_name', 'Unknown')}**:\n\n"
#         table_str += "| Nutrient       | Value |\n"
#         table_str += "|----------------|-------------------------------|\n"
#         for name, value in table:
#             table_str += f"| {name:<14} | {value} |\n"

#         return table_str

#     except requests.exceptions.RequestException as e:
#         return f"❌ Request failed: {str(e)}"
#     except Exception as e:
#         return f"❌ An unexpected error occurred: {str(e)}"



##################################################---WORKING--CODE---########################################################
# import requests

# APP_ID = "55b6e996"
# APP_KEY = "fa595b959fd39e454fa4627608bf20ab"  # Cleaned key

# def get_nutrition(food_item):
#     url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
#     headers = {
#         "x-app-id": APP_ID,
#         "x-app-key": APP_KEY,
#         "Content-Type": "application/json"
#     }
#     body = {
#         "query": food_item
#     }

#     try:
#         response = requests.post(url, headers=headers, json=body)
#         response.raise_for_status()
#         data = response.json()

#         if "foods" in data and len(data["foods"]) > 0:
#             food = data["foods"][0]

#             # Return all nutrient fields available
#             return {
#                 "food_name": food.get("food_name"),
#                 "serving_qty": food.get("serving_qty"),
#                 "serving_unit": food.get("serving_unit"),
#                 "serving_weight_grams": food.get("serving_weight_grams"),
#                 "calories": food.get("nf_calories"),
#                 "total_fat": food.get("nf_total_fat"),
#                 "saturated_fat": food.get("nf_saturated_fat"),
#                 "cholesterol": food.get("nf_cholesterol"),
#                 "sodium": food.get("nf_sodium"),
#                 "total_carbohydrate": food.get("nf_total_carbohydrate"),
#                 "dietary_fiber": food.get("nf_dietary_fiber"),
#                 "sugars": food.get("nf_sugars"),
#                 "protein": food.get("nf_protein"),
#                 "potassium": food.get("nf_potassium")
#             }
#         else:
#             return {"error": "No nutritional data found for the given item."}

#     except requests.exceptions.RequestException as e:
#         return {"error": f"Request failed: {str(e)}"}
#     except Exception as e:
#         return {"error": f"An unexpected error occurred: {str(e)}"}

#########################################################################################################################


# import requests

# APP_ID = "55b6e996"
# APP_KEY = "fa595b959fd39e454fa4627608bf20ab	—"

# def get_nutrition(food_item):
#     url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
#     headers = {
#         "x-app-id": APP_ID,
#         "x-app-key": APP_KEY,
#         "Content-Type": "application/json"
#     }
#     body = {
#         "query": food_item
#     }

#     response = requests.post(url, headers=headers, json=body)
#     data = response.json()

#     if "foods" in data and len(data["foods"]) > 0:
#         food = data["foods"][0]
#         return {
#             "calories": food.get("nf_calories"),
#             "protein": food.get("nf_protein"),
#             "carbs": food.get("nf_total_carbohydrate"),
#             "fat": food.get("nf_total_fat")
#         }
#     else:
#         return {"error": "Nutritional data not found from Nutritionix."}







# # def get_nutrition(food_item):
# #     sample_data = {
# #         "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10},
# #         "hamburger": {"calories": 354, "protein": 17, "carbs": 29, "fat": 21},
# #         "green_salad": {"calories": 152, "protein": 5, "carbs": 11, "fat": 10},
# #         "spaghetti_bolognese": {"calories": 221, "protein": 8, "carbs": 43, "fat": 1.3},
# #         "sushi": {"calories": 200, "protein": 6, "carbs": 28, "fat": 5},
# #         "french_fries": {"calories": 312, "protein": 3.4, "carbs": 41, "fat": 15}
# #     }
# #     key = food_item.lower().replace(" ", "_")
# #     return sample_data.get(key, {"error": "Nutritional data not found."})