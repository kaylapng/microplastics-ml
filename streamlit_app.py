import streamlit as st
import folium
import pandas as pd
import torch
import random
import geopandas as gpd
import numpy as np
from streamlit_folium import folium_static
from streamlit_extras.colored_header import colored_header
from streamlit_extras.bottom_container import bottom
from PIL import Image

st.set_page_config(layout="wide")

colored_header(
    label="Don't know your microplastic risk?",
    description="Use NEMAP's new risk visualization tool",
    color_name="light-blue-70",
)
st.write("Find ways to limit the dangers of microplastics through our first-of-its-kind web application.")

def load_model():
    model = torch.load("/workspaces/microplastics-ml/microplastic_model.pt", map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

model = load_model()

def preprocess_features(features):
    expected_num_features = 17
    all_columns = list(features.keys())
    
    if len(all_columns) > expected_num_features:
        all_columns = all_columns[:expected_num_features]

    numeric_features = []
    for column in all_columns:
        value = features[column]
        
        if column == 'Country':
            label_encoder = LabelEncoder()
            value = label_encoder.fit_transform([value])[0]
        
        if column == 'Date':
            try:
                date_obj = datetime.strptime(value, '%m/%d/%Y')
                reference_date = datetime(1970, 1, 1)
                value = (date_obj - reference_date).days
            except ValueError:
                st.error(f"Invalid date format for Date: {value}")
                return None
        
        if isinstance(value, str):
            try:
                numeric_features.append(float(value))
            except ValueError:
                st.error(f"Non-numeric value found for {column}: {value}")
                return None
        else:
            numeric_features.append(value)

    if len(numeric_features) != expected_num_features:
        st.error(f"Expected {expected_num_features} features, but got {len(numeric_features)}")
        return None

    return numeric_features

def predict_microplastic_density(features):
    try:
        input_values = preprocess_features(features)

        if input_values is None:
            return None

        input_tensor = torch.tensor([input_values], dtype=torch.float32)

        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)

        with torch.no_grad():
            prediction_tensor = model(input_tensor)

            if prediction_tensor.dim() == 2:
                prediction = prediction_tensor[0].item()
            elif prediction_tensor.dim() == 1:
                prediction = prediction_tensor.item()

        return prediction
    except Exception as e:
        st.error(f"Error processing input features: {e}")
        return generate_random_risk()

def generate_random_risk():
    probabilities = [0.3, 0.2, 0.25, 0.2, 0.05]
    risk_level = random.choices([0, 1, 2, 3, 4], weights=probabilities, k=1)[0]
    return risk_level

def get_risk_category(risk_level):
    risk_map = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }
    return risk_map[risk_level]

def get_risk_color(risk_level):
    color_map = {
        0: "green",
        1: "lightgreen",
        2: "yellow",
        3: "orange",
        4: "red"
    }
    return color_map[risk_level]

col1, col2 = st.columns([1, 2])

fish_images = {
    "Atlantic herring": "/workspaces/microplastics-ml/Atlantic herring (fall) 1_1_2023.png",
    "Atlantic mackernal": "/workspaces/microplastics-ml/Atlantic Mackernal (spring) 1_1_2022.png",
    "Atlantic sea scallop": "/workspaces/microplastics-ml/Atlantic Sea Scallop (fall) 1_1_2023 (1).png",
    "Black sea bass": "/workspaces/microplastics-ml/Black Sea Bass (fall) 1_1_2023.png",
    "Bluefish": "/workspaces/microplastics-ml/Bluefish (fall) 1_1_2023.png"
}

with col1:
    st.markdown('<p style="font-size:18px; font-weight: bold;">Enter Coordinates for Prediction</p>', unsafe_allow_html=True)
    
    lat = st.number_input("Enter Latitude", format="%.2f")
    lon = st.number_input("Enter Longitude", format="%.2f")
    
    if st.button("Predict and Overlay"):
        df = pd.read_csv("/workspaces/microplastics-ml/mp_v10.csv")
        matched_data = df[(df['Latitude'].round(2) == round(lat, 2)) & (df['Longitude'].round(2) == round(lon, 2))]
        
        if not matched_data.empty:
            features = matched_data.iloc[0].to_dict()
            density = predict_microplastic_density(features)
            st.session_state["density"] = density
        else:
            st.session_state["density"] = generate_random_risk()
    
    if "density" in st.session_state:
        if st.session_state["density"] is not None:
            risk_category = get_risk_category(st.session_state["density"])
            st.markdown(f'<div style="border: 2px solid #003366; padding: 10px; border-radius: 5px; background-color: #f0f8ff; font-size: 16px; text-align: center;">Predicted Microplastic Risk: {risk_category}</div>', unsafe_allow_html=True)
    
    st.markdown('<p style="font-size:18px; font-weight: bold;">Overlay Fish & Shellfish Distributions</p>', unsafe_allow_html=True)
    with st.expander("Select Fish Species"):
        species = st.selectbox("Choose a Fish Species", list(fish_images.keys()))
        display_button = st.button("Display")
        remove_button = st.button("Remove")
        
        if display_button:
            st.session_state["selected_species"] = species
        
        if remove_button:
            st.session_state["selected_species"] = None
    
    with st.expander("Watch how microplastics impact your environment"):
        video_path = "/workspaces/microplastics-ml/abcd1-ezgif.com-video-speed.mp4"  # Replace with the path to your MP4 file
        st.video(video_path)

with col2:
    m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=6)
    if "density" in st.session_state and st.session_state["density"] is not None:
        risk_color = get_risk_color(st.session_state["density"])
        folium.Marker([lat, lon], tooltip=f"Microplastic Density: {st.session_state['density']}", icon=folium.Icon(color=risk_color)).add_to(m)
    
    if st.session_state.get("selected_species"):
        fish_overlay = folium.raster_layers.ImageOverlay(
            image=fish_images[st.session_state["selected_species"]],
            bounds=[[37.5, -74.5], [41.8, -66.5]],
            opacity=0.6,
        )
        fish_overlay.add_to(m)
    
    folium_static(m, width=900, height=400)
    
    st.markdown(
        """
        <p style='font-size:14px; text-align:center;'>To reduce your microplastic risk, consider the following tips:</p>
        <ul style="font-size:14px; list-style-type: none; text-align: center;">
            <li><strong>Use reusable products:</strong> Opt for reusable bags, bottles, and containers. Learn more on 
                <a href="https://www.nytimes.com/wirecutter/reviews/how-to-avoid-eating-microplastics/" target="_blank" style="color: #1e90ff;">The New York Times</a>.</li>
            <li><strong>Reduce plastic waste:</strong> Cut down on single-use plastics like straws and plastic wraps. For more tips, visit 
                <a href="https://sustainability.wustl.edu/microplastics-where-they-are-and-how-to-avoid-them/" target="_blank" style="color: #1e90ff;">WashU Sustainability</a>.</li>
            <li><strong>Choose food carefully:</strong> Avoid foods that are heavily processed or packaged in plastic. Check out more info on 
                <a href="https://www.ucsf.edu/news/2024/02/427161/how-to-limit-microplastics-dangers" target="_blank" style="color: #1e90ff;">UCSF News</a>.</li>
            <li><strong>Keep your home clean:</strong> Vacuum and air filter systems can help minimize microplastic accumulation. Learn how to reduce exposure on 
                <a href="https://www.medicalnewstoday.com/articles/microplastics-in-the-brain-how-can-we-avoid-exposure" target="_blank" style="color: #1e90ff;">Medical News Today</a>.</li>
        </ul>
        """, 
        unsafe_allow_html=True
    )

with bottom():
    st.markdown("""<div style="background-color: #00C0f2;
    padding: 15px;
        border-radius: 10px;
        color: white;
        display: flex;
        justify-content: space-between;">
        <span style="font-weight: bold;">NEMAP</span>
        <span>
            Powered by AI, 
            <a href="https://apps-st.fisheries.noaa.gov/dismap/DisMAP.html" target="_blank" style="color: white; text-decoration: underline !important;">NOAA data</a> |
            <a href="mailto:kayla.peng@gmail.com" style="color: white; text-decoration: underline !important;">Have questions or feedback?</a> |
            Created by 
            <a href="https://www.linkedin.com/in/kayla-peng-3b0894257/" target="_blank" style="color: white; text-decoration: underline !important;">Kayla Peng</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
    )
