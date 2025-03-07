from flask import Flask, render_template, request, jsonify, send_from_directory, session
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import json
import requests
from datetime import datetime
import uuid

app = Flask(__name__, 
    static_url_path='', 
    static_folder='static',
    template_folder='templates')
app.secret_key = os.urandom(24)  # Secret key for sessions

# Initialize global variables
llm = None
user_preferences = {}  # Store user preferences

# Define seasons in Southern Africa
def get_current_season():
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return "autumn"
    elif 6 <= current_month <= 8:
        return "winter"
    elif 9 <= current_month <= 11:
        return "spring"
    else:
        return "summer"

# Define common crops by season in Southern Africa
seasonal_crops = {
    "summer": ["maize", "sorghum", "millet", "groundnuts", "cotton", "soybeans", "sunflower", "tobacco", "vegetables"],
    "autumn": ["winter wheat", "barley", "potatoes", "vegetable harvest", "land preparation"],
    "winter": ["wheat", "barley", "oats", "peas", "leafy greens", "onions", "garlic"],
    "spring": ["maize preparation", "tobacco seedbeds", "cotton preparation", "vegetable planting", "soil preparation"]
}

# Define the LLM - Now Using Groq
def initialize_llm():
    """Initializes the Groq language model with Llama."""
    
    api_key = "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"
    
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize Groq with Llama
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,  # Slightly increased temperature for more natural farming advice
        max_tokens=4096,
    )
    return llm

# Response templates for common questions
response_templates = {
    "greeting": "Hello! I'm Mudhumeni AI, your Southern African farming assistant. How can I help with your agricultural questions today?",
    "weather_general": "Weather patterns are critical for farming in Southern Africa. Currently, we're in {season} season. Would you like specific weather information for your region?",
    "unknown": "I don't have enough information to answer that question specifically for Southern Africa. However, as a general agricultural principle, {general_advice}. Would you like me to provide more general information on this topic?",
    "crop_general": "In Southern Africa, popular crops include maize, tobacco, cotton, wheat, sorghum, and various fruits and vegetables. Each has specific requirements for soil, water, and climate. What crop are you interested in cultivating?",
    "current_season": "We're currently in {season} season in Southern Africa. This is typically a good time for {seasonal_activities}. The main crops to focus on now are: {crops}."
}

# Generate system prompt based on user context and preferences
def generate_system_prompt(user_id=None):
    season = get_current_season()
    seasonal_focus = ", ".join(seasonal_crops[season][:5])
    
    user_location = "Southern Africa"
    farming_type = "various types of"
    
    # If we have user preferences, personalize the prompt
    if user_id and user_id in user_preferences:
        prefs = user_preferences[user_id]
        if prefs.get("location"):
            user_location = prefs["location"]
        if prefs.get("farming_type"):
            farming_type = prefs["farming_type"]
    
    system_prompt = f"""You are Mudhumeni AI, an AI-powered farming guide specifically designed for farmers in Southern Africa.
    Your purpose is to provide region-specific agricultural advice, considering the unique climate, soil conditions, and farming practices of Southern Africa.
    
    Current context: It is currently {season} season in Southern Africa, which is typically good for {seasonal_focus}.
    
    Focus on crops that are common in the region such as maize, tobacco, cotton, wheat, sorghum, millet, groundnuts, soybeans, and various fruits and vegetables.
    Provide practical advice on planting, irrigation, pest control, harvesting, and sustainability that is appropriate for the Southern African context.
    Consider the specific challenges of the region such as drought, variable rainfall, and resource constraints.
    Be knowledgeable about both commercial and subsistence farming practices relevant to the region.
    
    The user is likely from {user_location} and engaged in {farming_type} farming.
    
    When you don't know something specific to the region, acknowledge this limitation and provide general agricultural principles that could be adapted.
    Always be respectful, helpful, and mindful of the diverse farming conditions across Zimbabwe, South Africa, Botswana, Zambia, Mozambique, Namibia, and other Southern African countries.
    
    Example exchanges:
    
    User: When should I plant maize?
    Assistant: In most parts of Southern Africa, maize is typically planted at the onset of the rainy season, usually from mid-November to mid-December. However, this varies by your specific location. In Zimbabwe's highveld, for example, mid-November is ideal, while in South Africa's Free State, early December might be better. The key is to plant after receiving 15-20mm of rainfall when soil moisture is adequate. Are you planting for commercial or subsistence purposes?
    
    User: How do I deal with fall armyworm?
    Assistant: Fall armyworm has become a significant pest in Southern Africa since 2016. For immediate control: 1) Scout your fields early morning or late afternoon when the larvae are active; 2) For small-scale farmers, consider physical removal and destruction of egg masses and larvae; 3) Apply registered pesticides like Indoxacarb or Spinetoram, but rotate chemicals to prevent resistance; 4) Consider biological controls like Bacillus thuringiensis (Bt) products which are effective and environmentally friendly. Prevention strategies include early planting, crop rotation, and intercropping with legumes like cowpeas. What size is your farm, so I can suggest the most appropriate control method?
    
    User: What's a drought-resistant crop I can grow?
    Assistant: For Southern African drought conditions, sorghum (mapfunde/amazimba) is an excellent choice. It requires 30% less water than maize and can remain dormant during dry spells. Small grains like pearl millet (mhunga/amabele) and finger millet (rukweza/uphoko) are also highly drought-tolerant and nutritious. For legumes, cowpeas (nyemba) and pigeon peas (nyimo) perform well with minimal rainfall. Drought-tolerant maize varieties developed specifically for Southern Africa are available through seed companies like SeedCo and Pannar. Would you like specific information on cultivation practices for any of these crops?
    """
    return system_prompt

# Chatbot logic
def chatbot_response(user_input, user_id=None):
    global llm
    try:
        if not user_input.strip():
            return "Please enter a valid question."
        
        # Check for special command to set preferences
        if user_input.lower().startswith("set location:"):
            location = user_input[13:].strip()
            if user_id not in user_preferences:
                user_preferences[user_id] = {}
            user_preferences[user_id]["location"] = location
            return f"Thank you! I've noted that you're farming in {location}. This will help me provide more relevant advice."
        
        if user_input.lower().startswith("set farming:"):
            farming_type = user_input[12:].strip()
            if user_id not in user_preferences:
                user_preferences[user_id] = {}
            user_preferences[user_id]["farming_type"] = farming_type
            return f"Thank you! I've noted that you're involved in {farming_type} farming. I'll tailor my advice accordingly."
        
        # Check for season inquiry
        if "season" in user_input.lower() and ("current" in user_input.lower() or "now" in user_input.lower()):
            season = get_current_season()
            crops = ", ".join(seasonal_crops[season])
            seasonal_activities = "land preparation and planting" if season == "spring" else \
                                 "growing and maintenance" if season == "summer" else \
                                 "harvesting and storage" if season == "autumn" else "planning and soil preparation"
            return response_templates["current_season"].format(season=season, seasonal_activities=seasonal_activities, crops=crops)
        
        # Get conversation history if available
        chat_history = session.get('chat_history', [])
        history_text = ""
        
        # Only use the last 5 exchanges to keep context relevant
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for exchange in recent_history:
            if exchange["role"] == "user":
                history_text += f"User: {exchange['content']}\n"
            else:
                history_text += f"Assistant: {exchange['content']}\n"
        
        # Generate system prompt based on user context
        system_prompt = generate_system_prompt(user_id)
        
        # Construct the full prompt with system message, history, and user input
        if history_text:
            full_prompt = f"{system_prompt}\n\nPrevious conversation:\n{history_text}\n\nUser: {user_input}\nAssistant:"
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
            
        # Call the LLM with the full context
        response = llm.invoke(full_prompt)
        
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Network error in chatbot response: {str(e)}")
        return "I'm having trouble connecting to my knowledge database. Please check your internet connection and try again in a moment."
    except ValueError as e:
        print(f"Value error in chatbot response: {str(e)}")
        return "I couldn't understand a value in your question. Please try rephrasing with more specific measurements or terms."
    except Exception as e:
        print(f"Error in chatbot response: {str(e)}")
        return "I apologize, but I encountered an error. Please try asking your question differently or try again later."

# Initialize outside main to be global
print("Initializing Mudhumeni AI Chatbot......")
llm = initialize_llm()

# Landing page route
@app.route('/')
@app.route('/landing')
def landing():
    return render_template('landing.html')

# Chatbot route
@app.route('/chatbot')
def chatbot():
    # Generate a user ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Initialize chat history if needed
    if 'chat_history' not in session:
        session['chat_history'] = []
        
    return render_template('index.html')

# Crop recommendation route
@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

# Predict crop route
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Load the model and scalers
        model = pickle.load(open('model.pkl', 'rb'))
        minmaxscaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
        standscaler = pickle.load(open('standscaler.pkl', 'rb'))
        
        # Get data from form
        data = request.form
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        mx_features = minmaxscaler.transform(features)
        sc_mx_features = standscaler.transform(mx_features)
        prediction = model.predict(sc_mx_features)
        
        # Crop dictionary
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
            6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
            11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate',
            15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans',
            19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
        }
        
        # Add Southern African context to the prediction
        predicted_crop = crop_dict[prediction[0]]
        
        # Get current season
        season = get_current_season()
        
        # Add seasonal advice
        seasonal_advice = ""
        if predicted_crop in seasonal_crops[season]:
            seasonal_advice = f"Good choice! {predicted_crop.title()} is well-suited for the current {season} season in Southern Africa."
        else:
            appropriate_season = next((s for s, crops in seasonal_crops.items() if predicted_crop in crops), None)
            if appropriate_season:
                seasonal_advice = f"Note: {predicted_crop.title()} is typically more suited for the {appropriate_season} season in Southern Africa. Consider planning accordingly."
        
        return jsonify({
            'success': True, 
            'prediction': predicted_crop,
            'season': season,
            'seasonal_advice': seasonal_advice
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Chat response route
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form["user_input"]
    user_id = session.get('user_id')
    
    # Get existing chat history or initialize
    chat_history = session.get('chat_history', [])
    
    # Add user input to history
    chat_history.append({"role": "user", "content": user_input})
    
    # Get response from chatbot
    response = chatbot_response(user_input, user_id)
    
    # Add response to history
    chat_history.append({"role": "assistant", "content": response})
    
    # Save updated history to session
    session['chat_history'] = chat_history
    
    return jsonify({"response": response, "chat_history": chat_history})

# About route
@app.route("/about")
def about():
    return render_template("about.html")

# Weather route
@app.route('/weather')
def weather_page():
    return render_template('weather.html')

# Add a route to handle weather data requests
@app.route('/get_weather/<province>')
def get_weather(province):
    try:
        # Try to get real weather data from OpenWeatherMap API (you'll need an API key)
        api_key = ""  # Add your OpenWeatherMap API key here if available
        
        if api_key and province:
            # Map province names to main cities for weather lookup
            province_cities = {
                'harare': 'Harare',
                'bulawayo': 'Bulawayo',
                'manicaland': 'Mutare',
                'mashonaland_central': 'Bindura',
                'mashonaland_east': 'Marondera',
                'mashonaland_west': 'Chinhoyi',
                'masvingo': 'Masvingo',
                'matabeleland_north': 'Hwange',
                'matabeleland_south': 'Gwanda',
                'midlands': 'Gweru'
            }
            
            city = province_cities.get(province.lower(), province)
            
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                response = requests.get(url)
                data = response.json()
                
                if response.status_code == 200:
                    return jsonify({
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'rainfall_chance': data.get('rain', {}).get('1h', 0) * 100 if 'rain' in data else 0,
                        'wind_speed': data['wind']['speed'],
                        'description': data['weather'][0]['description'],
                        'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
                    })
            except:
                # If API call fails, fall back to mock data
                pass
        
        # Mock data as fallback or if no API key is provided
        season = get_current_season()
        weather_data = {
            'harare': {
                'temperature': 25 if season in ['spring', 'summer'] else 18,
                'humidity': 65 if season in ['summer', 'autumn'] else 45,
                'rainfall_chance': 20 if season == 'summer' else 5,
                'wind_speed': 12,
                'description': f'Typical {season} conditions for Harare region. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'bulawayo': {
                'temperature': 28 if season in ['spring', 'summer'] else 20,
                'humidity': 55 if season in ['summer', 'autumn'] else 35,
                'rainfall_chance': 10 if season == 'summer' else 2,
                'wind_speed': 15,
                'description': f'Typical {season} conditions for Bulawayo region. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'manicaland': {
                'temperature': 22 if season in ['spring', 'summer'] else 15,
                'humidity': 75 if season in ['summer', 'autumn'] else 55,
                'rainfall_chance': 30 if season == 'summer' else 10,
                'wind_speed': 8,
                'description': f'Typical {season} conditions for Manicaland region. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'mashonaland_central': {
                'temperature': 26 if season in ['spring', 'summer'] else 19,
                'humidity': 60 if season in ['summer', 'autumn'] else 40,
                'rainfall_chance': 15 if season == 'summer' else 5,
                'wind_speed': 10,
                'description': f'Typical {season} conditions for Mashonaland Central. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'mashonaland_east': {
                'temperature': 24 if season in ['spring', 'summer'] else 17,
                'humidity': 62 if season in ['summer', 'autumn'] else 42,
                'rainfall_chance': 25 if season == 'summer' else 8,
                'wind_speed': 9,
                'description': f'Typical {season} conditions for Mashonaland East. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'mashonaland_west': {
                'temperature': 27 if season in ['spring', 'summer'] else 20,
                'humidity': 58 if season in ['summer', 'autumn'] else 38,
                'rainfall_chance': 12 if season == 'summer' else 3,
                'wind_speed': 11,
                'description': f'Typical {season} conditions for Mashonaland West. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'masvingo': {
                'temperature': 29 if season in ['spring', 'summer'] else 22,
                'humidity': 50 if season in ['summer', 'autumn'] else 30,
                'rainfall_chance': 5 if season == 'summer' else 1,
                'wind_speed': 14,
                'description': f'Typical {season} conditions for Masvingo region. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'matabeleland_north': {
                'temperature': 30 if season in ['spring', 'summer'] else 23,
                'humidity': 45 if season in ['summer', 'autumn'] else 25,
                'rainfall_chance': 5 if season == 'summer' else 1,
                'wind_speed': 16,
                'description': f'Typical {season} conditions for Matabeleland North. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'matabeleland_south': {
                'temperature': 31 if season in ['spring', 'summer'] else 24,
                'humidity': 40 if season in ['summer', 'autumn'] else 20,
                'rainfall_chance': 2 if season == 'summer' else 0,
                'wind_speed': 18,
                'description': f'Typical {season} conditions for Matabeleland South. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            },
            'midlands': {
                'temperature': 26 if season in ['spring', 'summer'] else 19,
                'humidity': 55 if season in ['summer', 'autumn'] else 35,
                'rainfall_chance': 15 if season == 'summer' else 5,
                'wind_speed': 12,
                'description': f'Typical {season} conditions for Midlands region. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            }
        }
        
        # Return data for the requested province or a default response
        if province.lower() in weather_data:
            return jsonify(weather_data[province.lower()])
        else:
            return jsonify({
                'temperature': 26 if season in ['spring', 'summer'] else 19,
                'humidity': 60 if season in ['summer', 'autumn'] else 40,
                'rainfall_chance': 15 if season == 'summer' else 5,
                'wind_speed': 10,
                'description': f'Weather data not available for this specific region. Showing average Southern African {season} conditions. {season.capitalize()} crops to consider: {", ".join(seasonal_crops[season][:3])}',
                'last_updated': datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            })
    except Exception as e:
        return jsonify({'error': str(e)})

# Add a route for multilingual support - translate response to a supported language
@app.route('/translate', methods=['POST'])
def translate_response():
    try:
        data = request.json
        text = data.get('text', '')
        target_language = data.get('language', 'en')  # Default to English
        
        # Supported languages in Southern Africa
        supported_languages = {
            'en': 'English',
            'af': 'Afrikaans',
            'st': 'Sesotho',
            'tn': 'Setswana',
            'xh': 'isiXhosa',
            'zu': 'isiZulu',
            'sn': 'Shona',
            'nd': 'Ndebele',
            'sw': 'Swahili',
            'pt': 'Portuguese'  # For Mozambique, Angola
        }
        
        # For now, this is a mock translation as integrating a real translation API would require additional setup
        # You could integrate Google Translate API or similar service here
        
        if target_language != 'en' and target_language in supported_languages:
            # This is a mock - in production, you would call a translation API
            translated_text = f"[This would be translated to {supported_languages[target_language]}]: {text}"
            return jsonify({'translated_text': translated_text, 'language': target_language})
        else:
            return jsonify({'translated_text': text, 'language': 'en'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Add a route to set user preferences
@app.route('/set_preferences', methods=['POST'])
def set_user_preferences():
    try:
        data = request.json
        user_id = session.get('user_id')
        
        if not user_id:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
        
        if user_id not in user_preferences:
            user_preferences[user_id] = {}
        
        # Update preferences
        if 'location' in data:
            user_preferences[user_id]['location'] = data['location']
        
        if 'farming_type' in data:
            user_preferences[user_id]['farming_type'] = data['farming_type']
        
        return jsonify({
            'success': True, 
            'message': 'Preferences updated successfully',
            'preferences': user_preferences[user_id]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Clear chat history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    return jsonify({'success': True, 'message': 'Chat history cleared'})

if __name__ == '__main__':
    app.run(debug=True)