import json
import os

# --- Constants (Keep as before) ---
general_workout_tips = [
    "Always warm up for 5-10 minutes before exercising and cool down for 5-10 minutes afterward.",
    "Listen to your body; stop or modify if you feel sharp or persistent pain.",
    "Stay hydrated by drinking water throughout the day, especially before, during, and after exercise.",
    "Consult your doctor or a qualified healthcare professional before starting any new exercise program, especially if you have underlying health conditions.",
    "Consistency is key! Aim to stick to the plan as best as possible for optimal results.",
    "Wear appropriate footwear and clothing for your chosen activities.",
    "Focus on proper form to maximize effectiveness and minimize risk of injury."
]

general_diet_tips = [
    "Focus on whole, unprocessed foods like fruits, vegetables, whole grains, lean proteins, and healthy fats.",
    "Drink plenty of water throughout the day (aim for 8 glasses or as advised by your doctor).",
    "Limit sugary drinks (soda, juice, sweetened teas) and snacks (candy, pastries).",
    "Pay attention to portion sizes to manage calorie intake.",
    "Read food labels to understand ingredients, serving sizes, calories, sugar, sodium, and fat content.",
    "Prioritize fiber-rich foods (vegetables, fruits, legumes, whole grains) to aid digestion and blood sugar control.",
    "Choose healthy fats (avocado, nuts, seeds, olive oil) over saturated and trans fats.",
    "Limit sodium intake by avoiding processed foods and adding less salt during cooking.",
    "Consult a registered dietitian or your doctor for personalized dietary advice tailored to your specific needs and health conditions."
]

diet_plan_reasoning = {
    "high_bmi": "This diet plan typically focuses on a moderate calorie deficit to support healthy weight management. It emphasizes nutrient-dense foods and adequate protein to promote satiety (feeling full) while controlling overall energy intake.",
    "high_bp": "This diet plan emphasizes limiting sodium intake and including foods rich in potassium, magnesium, and calcium (like fruits, vegetables, whole grains, lean protein) to help manage blood pressure, following principles similar to the DASH diet.",
    "cholesterol": "This diet plan focuses on managing blood lipid levels by limiting saturated and trans fats, while emphasizing sources of unsaturated fats (like olive oil, avocados, nuts, seeds) and soluble fiber (like oats, beans, apples), which can help lower LDL ('bad') cholesterol.",
    "insulin_resistance": "This diet plan aims to improve the body's sensitivity to insulin. It prioritizes low-glycemic index carbohydrates, consistent meal timing, adequate fiber, lean protein, and healthy fats to help stabilize blood sugar levels.",
    "glucose_levels": "This diet plan focuses on managing blood sugar levels by controlling carbohydrate intake (emphasizing low-glycemic index and portion control), ensuring high fiber consumption, and balancing meals with protein and healthy fats. Limiting added sugars and refined carbs is key.",
    "other": "This general healthy eating plan is recommended as the primary risk factor was 'life style ' . It emphasizes balanced nutrition with whole foods, adequate fiber, lean protein, healthy fats, and limiting processed foods and added sugars.",
    "low_risk_default": "As your risk is assessed as low, this plan focuses on maintaining a generally healthy and balanced diet with a variety of whole foods to support overall well-being and continued prevention."
}

specific_risk_tips = {
    "High BMI": "🎯 Focus on portion control and choose whole foods to help manage weight.",
    "High Cholesterol": "🥑 Prioritize healthy fats (like avocados, nuts, olive oil) and soluble fiber (like oats, beans), and limit saturated/trans fats.",
    "High BP": "🧂 Be mindful of sodium intake; choose fresh foods over highly processed options and limit added salt.",
    "Glucose Levels": "⚖️ Choose carbohydrates with a lower glycemic index and pair carbs with protein or healthy fats to help manage blood sugar.",
    "Insulin Resistance": "⏱️ Consistent meal timing and balanced meals focusing on fiber, lean protein, and healthy fats can help improve insulin sensitivity.",
    "Family History": "🩺 Regular check-ups with your doctor are especially important given your family history."
}


# --- Helper Functions (Keep as before or with minor emoji additions) ---
def get_age_group(age):
    if 18 <= age <= 39: return "18-39"
    if 40 <= age <= 64: return "40-64"
    if age >= 65: return "65-80+"
    print(f"Warning: Age {age} is outside defined ranges (18+).")
    return None


def get_primary_risk_key_for_plan(risk_factors_list):
    predefined_plan_risks_map = {
        "high bmi": "high_bmi", "high bp": "high_bp", "cholesterol": "cholesterol",
        "insulin resistance": "insulin_resistance", "glucose levels": "glucose_levels",
    }
    json_plan_keys = ["high_bmi", "high_bp", "cholesterol", "insulin_resistance", "glucose_levels", "other"]
    if not risk_factors_list or not isinstance(risk_factors_list, list) or not risk_factors_list[0]:
        return "other", "Other (Not specified)"
    primary_risk_original = risk_factors_list[0].strip()
    primary_risk_lower = primary_risk_original.lower()
    plan_key = predefined_plan_risks_map.get(primary_risk_lower)
    if plan_key and plan_key in json_plan_keys:
        return plan_key, primary_risk_original
    else:
        print(
            f"Warning: Primary risk factor '{primary_risk_original}' not recognized for specific plan lookup. Using 'other' category for plan.")
        return "other", primary_risk_original


def clean_input_key(input_string):
    if not isinstance(input_string, str): return None
    return input_string.lower().replace(" ", "_")


def load_json_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: JSON file not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)  # Added encoding
        return data
    except Exception as e:
        print(f"Error loading or parsing JSON file {filepath}: {e}")
        return None


def format_meal_plan_as_markdown_table(meal_plan_list):
    if not meal_plan_list or not isinstance(meal_plan_list, list):
        return "\n🤷 No specific meal examples available for this plan.\n"
    # ... (table formatting as before) ...
    markdown_string = "\n| 🗓️ Day | 🍳 Breakfast | 🥗 Lunch | 🍲 Dinner | 🍎 Snacks |\n"  # Added Emojis
    markdown_string += "|---|---|---|---|---|\n"
    for day_meal in meal_plan_list:
        day = day_meal.get("Day", "?")
        breakfast = day_meal.get("Breakfast", "N/A")
        lunch = day_meal.get("Lunch", "N/A")
        dinner = day_meal.get("Dinner", "N/A")
        snacks = day_meal.get("Snacks", "N/A")
        markdown_string += f"| {day} | {breakfast} | {lunch} | {dinner} | {snacks} |\n"
    return markdown_string + "\n"


def format_workout_plan_as_markdown_table(workout_plan_list):
    if not workout_plan_list or not isinstance(workout_plan_list, list):
        return "\n🤷 No specific workout schedule available for this plan.\n"
    # ... (table formatting as before) ...
    markdown_string = "\n| 🗓️ DAY | 🏃 CV (Cardiovascular) | 💪 STR (Strength Training) | 🧘 FLX (Flexibility) | ⏱️ DURATION | 📈 FREQUENCY/INTENSITY | 📝 NOTES |\n"  # Added Emojis
    markdown_string += "|---|---|---|---|---|---|---|\n"
    for day_workout in workout_plan_list:
        day = day_workout.get("Day", "?")
        cv = day_workout.get("CV", "N/A")
        str_ex = day_workout.get("STR", "N/A")
        flx = day_workout.get("FLX", "N/A")
        duration = day_workout.get("Duration", "N/A")
        frequency_intensity = day_workout.get("Frequency/Intensity", "N/A")
        notes = day_workout.get("Notes", "N/A")
        markdown_string += f"| {day} | {cv} | {str_ex} | {flx} | {duration} | {frequency_intensity} | {notes} |\n"
    return markdown_string + "\n"


def format_tips_as_markdown_bullets(tips_list, emoji="💡"):  # Default emoji
    if not tips_list or not isinstance(tips_list, list):
        return ""
    return "\n".join([f"* {emoji} {tip}" for tip in tips_list]) + "\n"


# --- Main Plan Generation Function ---
def generate_plan(risk_level, gender, age, risk_factors_list, activity_level, diet_habits):
    # ... (Input Validations and JSON loading - keep as is) ...
    risk_level_str = risk_level.strip().capitalize() if isinstance(risk_level, str) else "Unknown"
    gender_str = gender.strip().lower() if isinstance(gender, str) else None
    activity_level_key = clean_input_key(activity_level)
    diet_habits_key = clean_input_key(diet_habits)

    if gender_str not in ["male", "female"]: return "Error: Invalid gender provided."
    if not isinstance(age, int) or age < 18: return "Error: Invalid age provided (must be 18+)."
    age_group = get_age_group(age)
    if not age_group: return "Error: Could not determine age group for the provided age."
    valid_activities = ["active", "moderately_active", "non_active"]
    valid_diets = ["high_sugar_intake", "processed_food", "balanced_diet"]
    if activity_level_key not in valid_activities:
        print(
            f"Warning: Activity level '{activity_level}' (key: '{activity_level_key}') not in {valid_activities}. Plan lookup might fail if JSON is strict.")
    if diet_habits_key not in valid_diets:
        print(
            f"Warning: Diet habit '{diet_habits}' (key: '{diet_habits_key}') not in {valid_diets}. Plan lookup might fail if JSON is strict.")

    high_diet_data = load_json_data('high_diet_plans.json')
    high_workout_data = load_json_data('high_workout_plans.json')
    low_diet_data = load_json_data('low_diet_plans.json')
    low_workout_data = load_json_data('low_workout_plans.json')

    if risk_level_str == "High" and (not high_diet_data or not high_workout_data):
        return "Error: Could not load necessary high-risk plan data files. Ensure they exist and are valid."
    if risk_level_str == "Low" and (not low_diet_data or not low_workout_data):
        return "Error: Could not load necessary low-risk plan data files. Ensure they exist and are valid."

      # Consider passing actual name if available

    # Using a list to build the plan parts, then join with "\n\n" for better spacing
    plan_parts = []


    plan_parts.append(f"Your T2D risk assessment indicates: **{risk_level_str}**.")
    plan_parts.append(
        f"Based on your profile (Age: {age}, Gender: {gender_str.capitalize()}, Activity: {activity_level}, Diet: {diet_habits}), here is your suggested plan:")
    plan_parts.append("---")  # Markdown horizontal rule

    if risk_level_str == "High":
        primary_risk_plan_key, primary_risk_original_display = get_primary_risk_key_for_plan(risk_factors_list)
        if risk_factors_list and risk_factors_list[0]:
            plan_parts.append(f"**🔍 Identified Primary Risk Factors:** {', '.join(risk_factors_list)}")
        else:
            plan_parts.append(f"**🔍 Primary Risk Factor:** {primary_risk_original_display}")

        plan_parts.append("### 🍏 Dietary Plan")
        reasoning = diet_plan_reasoning.get(primary_risk_plan_key, diet_plan_reasoning["other"])
        plan_parts.append(f"**🧠 Reasoning:** {reasoning}")

        diet_plan = high_diet_data.get(gender_str, {}).get(age_group, {}).get(primary_risk_plan_key, {}).get(
            diet_habits_key)
        if diet_plan:
            plan_parts.append(
                f"**🎯 Target Calories:** {diet_plan.get('calories', 'N/A')}  \n**📊 Macronutrient Focus:** {diet_plan.get('macros', 'N/A')}")
            plan_parts.append(format_meal_plan_as_markdown_table(diet_plan.get('meal_plan')))
        else:
            plan_parts.append(
                f"🤷 Could not find a specific diet plan for your profile. Focusing on general high-risk tips.")

        plan_parts.append("**✨ Specific Diet Tips:**")
        specific_tips_to_show = []
        if risk_factors_list and risk_factors_list[0]:
            for factor in risk_factors_list:
                tip = specific_risk_tips.get(factor.strip())  # Emojis are now in specific_risk_tips
                if tip: specific_tips_to_show.append(tip)
        if specific_tips_to_show:
            plan_parts.append(
                format_tips_as_markdown_bullets(specific_tips_to_show, emoji=""))  # No extra emoji if already in tip
        else:
            plan_parts.append("* 💡 No specific factor-based tips available beyond general advice.")

        plan_parts.append("**🥗 General Healthy Diet Tips:**")
        plan_parts.append(format_tips_as_markdown_bullets(general_diet_tips, emoji="✅"))

        plan_parts.append("### 🏋️ WORKOUT Plan")
        # Get the list of workout days DIRECTLY for high-risk plans
        # assuming your JSON structure is: ...[risk_category_key][activity_level_key] -> LIST of days
        high_workout_schedule_list = high_workout_data.get(gender_str, {}).get(age_group, {}).get(primary_risk_plan_key,
                                                                                                  {}).get(
            activity_level_key)

        # Check if it's a non-empty list
        if high_workout_schedule_list and isinstance(high_workout_schedule_list, list):
            plan_parts.append(
                format_workout_plan_as_markdown_table(high_workout_schedule_list))  # Pass the list directly
        else:
            plan_parts.append(
                f"🤷 Could not find a specific workout plan for your profile. Focusing on general high-risk tips.")

        plan_parts.append("**🏃 General Workout Tips:**")
        plan_parts.append(format_tips_as_markdown_bullets(general_workout_tips, emoji="👍"))

    elif risk_level_str == "Low":
        plan_parts.append(
            f"Your risk is currently assessed as **low**, which is great! This plan focuses on maintaining healthy habits.")
        plan_parts.append("### 🍏 Dietary Plan")
        plan_parts.append(f"**🧠 Reasoning:** {diet_plan_reasoning['low_risk_default']}")
        low_diet_plan = low_diet_data.get(gender_str, {}).get(age_group, {}).get(diet_habits_key)
        if low_diet_plan:
            plan_parts.append(
                f"**🎯 Target Calories:** {low_diet_plan.get('calories', 'Not Specified')}  \n**📊 Macronutrient Focus:** {low_diet_plan.get('macros', 'Not Specified')}")
            plan_parts.append(format_meal_plan_as_markdown_table(low_diet_plan.get('meal_plan')))
        else:
            plan_parts.append(f"🤷 Could not find a specific low-risk diet example. Focusing on general low-risk tips.")
        plan_parts.append("**🥗 General Healthy Diet Tips (Maintain these habits):**")
        plan_parts.append(format_tips_as_markdown_bullets(general_diet_tips, emoji="✅"))


        plan_parts.append("### 🏋️ WORKOUT Plan")
        # Get the list of workout days directly
        low_workout_schedule_list = low_workout_data.get(gender_str, {}).get(age_group, {}).get(activity_level_key)

        # Check if it's a non-empty list
        if low_workout_schedule_list and isinstance(low_workout_schedule_list, list):
            plan_parts.append(format_workout_plan_as_markdown_table(low_workout_schedule_list)) # Pass the list directly
        else:
             plan_parts.append(f"🤷 Could not find a specific low-risk workout example. Focusing on general low-risk tips.")
        plan_parts.append("**🏃 General Workout Tips (Keep up the good work!):**")
        plan_parts.append(format_tips_as_markdown_bullets(general_workout_tips, emoji="👍"))
    else:
        return f"Error: Unknown risk level '{risk_level_str}'."

    plan_parts.append("### 🩺 Monitoring")
    monitoring_tips = ["Regular check-ups with your doctor are important for monitoring your health."]
    if risk_level_str == "High" and risk_factors_list and risk_factors_list[0]:
        if "Family History" in [rf.strip() for rf in risk_factors_list]:
            monitoring_tips.append(
                "Given your family history, discuss appropriate screening frequency with your doctor.")
        if any(f.strip() in [rf.strip() for rf in risk_factors_list] for f in
               ["High BP", "Glucose Levels", "High Cholesterol"]):
            monitoring_tips.append(
                "Follow your doctor's recommendations for monitoring blood pressure, blood sugar, and cholesterol levels.")
    plan_parts.append(format_tips_as_markdown_bullets(monitoring_tips, emoji="🗓️"))

    plan_parts.append("---")  # Markdown horizontal rule
    disclaimer_html = (
        "<div style='background-color: #444444; color: #FFD700; padding: 15px; border-radius: 8px; border: 1px solid #FFD700;'>"
        "<h4 style='margin-top:0; color: #FFFFFF;'>⚠️ IMPORTANT DISCLAIMER:</h4>"
        "<p>This generated plan provides general suggestions based on the information provided. "
        "<strong>It is NOT a substitute for professional medical advice.</strong></p>"
        "<p>Always consult with your doctor, a registered dietitian, or a qualified healthcare professional "
        "before making any changes to your diet, exercise routine, or health management plan, "
        "especially if you have any underlying health conditions.</p>"
        "</div>"
    )
    plan_parts.append(disclaimer_html)

    return "\n\n".join(plan_parts)  # Join parts with double newlines for spacing


# --- (Keep the __main__ example usage part for testing) ---
if __name__ == "__main__":
    # ... (your __main__ test code here, ensure JSON files are created or exist) ...
    # Create dummy JSON files for testing if they don't exist
    dummy_meal_plan = [
        {"Day": 1, "Breakfast": "Oats", "Lunch": "Salad", "Dinner": "Chicken", "Snacks": "Apple"},
        {"Day": 2, "Breakfast": "Eggs", "Lunch": "Soup", "Dinner": "Fish", "Snacks": "Nuts"}
    ]
    dummy_workout_plan_content = {"workout_schedule": [
        {"Day": 1, "CV": "Walk", "STR": "Squats", "FLX": "Stretch", "Duration": "30m",
         "Frequency/Intensity": "Moderate", "Notes": "Easy start"},
        {"Day": 2, "CV": "Bike", "STR": "Pushups", "FLX": "Yoga", "Duration": "45m", "Frequency/Intensity": "Vigorous",
         "Notes": "Full body"}
    ]}

    plans_to_create = {
        "low_diet_plans.json": {
            "female": {
                "18-39": {"balanced_diet": {"calories": "2000", "macros": "Balanced", "meal_plan": dummy_meal_plan}}},
        },
        "low_workout_plans.json": {
            "female": {"18-39": {"active": dummy_workout_plan_content}},
        },
        "high_diet_plans.json": {
            "female": {"40-64": {
                "high_bmi": {
                    "processed_food": {"calories": "1500", "macros": "High Protein", "meal_plan": dummy_meal_plan}},
                "other": {"high_sugar_intake": {"calories": "1700", "macros": "General Healthy",
                                                "meal_plan": dummy_meal_plan}}
            }}
        },
        "high_workout_plans.json": {
            "female": {"40-64": {
                "high_bmi": {"non_active": dummy_workout_plan_content},
                "other": {"active": dummy_workout_plan_content}
            }}
        }
    }
    for filename, content in plans_to_create.items():
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f: json.dump(content, f, indent=2)  # Added encoding
            print(f"Created dummy {filename}")

    print("--- Example 1: High Risk (High BMI) ---")
    user_input_high = {
        "risk_level": "High", "gender": "female", "age": 45,
        "risk_factors_list": ["High BMI", "Glucose Levels", "High BP"],
        "activity_level": "non_active", "diet_habits": "processed_food"
    }
    print(generate_plan(**user_input_high))

    print("\n\n--- Example 2: Low Risk ---")
    user_input_low = {
        "risk_level": "Low", "gender": "female", "age": 30,
        "risk_factors_list": [],
        "activity_level": "active", "diet_habits": "balanced_diet"
    }
    print(generate_plan(**user_input_low))