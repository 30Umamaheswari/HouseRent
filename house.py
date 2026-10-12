import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="House_Rent Calculator", layout="wide")

app_title = """
    color: lightseagreen;
    text-align: center;
    font-family: Serif;
    font-size: 50px;
"""
st.markdown(
    f'<h1 style="{app_title}">HouseHarbor: Your Rent Voyage Navigator</h1>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    # Created by train_model.py: encoders + model saved together in one pipeline
    with open('house_rent_pipeline.pkl', 'rb') as file:
        return pickle.load(file)


saved = load_model()
pipeline = saved['pipeline']
categories = saved['categories']

# Amenity checkbox label -> training column name
# (training data spells it 'Servent'; the label shows 'Servant')
amenities = {
    'Negotiable': 'Negotiable', 'Lift': 'Lift', 'Gym': 'Gym', 'Internet': 'Internet',
    'Air Conditioning': 'Air Conditioning', 'Club': 'Club', 'Intercom': 'Intercom',
    'Swimming Pool': 'Swimming Pool', 'Car Parking': 'Car Parking', 'Servant': 'Servent',
    'Security': 'Security', 'Shopping Center': 'Shopping Center', 'Gas Pipeline': 'Gas Pipeline',
    'Park': 'Park', 'Rain Water Harvesting': 'Rain Water Harvesting',
    'Sewage Treatment Plant': 'Sewage Treatment Plant', 'House Keeping': 'House Keeping',
    'Power Backup': 'Power Backup', 'Visitor Parking': 'Visitor Parking',
}

# Streamlit app Creation
col1, col2, col3 = st.columns([8, 4, 4], gap='large')

selected_house_type = col2.selectbox("Select House Type", categories['House Type'])
selected_lease_type = col2.selectbox("Select Lease Type", categories['Lease Type'])
selected_furniture = col2.selectbox("Select type of Furnishing", categories['Furnishing'])
selected_parking = col2.selectbox("Select type of Parking", categories['Parking'])
selected_house_facing = col2.selectbox("Choose House Facing", categories['House Facing'])
selected_water_supply = col2.selectbox("Choose Water Supply", categories['Water Supply'])
selected_building_type = col2.selectbox("Building Type", categories['Building Type'])

# house features
selected_property_size = col3.number_input("Property Size (sq ft)", min_value=100, max_value=10000, value=1000)
selected_property_age = col3.number_input("Property Age", min_value=0, max_value=100, value=5)
selected_no_of_bathroom = col3.number_input("Number of Bathroom", min_value=1, max_value=6, value=2)
selected_no_of_cupboard = col3.number_input("Number of Cup Board", min_value=0, max_value=15, value=2)
selected_floor_no = col3.number_input("Floor Number", min_value=0, max_value=20, value=1)
selected_no_of_floor = col3.number_input("Total Number of Floor", min_value=0, max_value=20, value=4)
selected_no_of_balconies = col3.number_input("Total Number of Balcony", min_value=0, max_value=6, value=1)

selected_amenities = col3.multiselect("Select amenities", list(amenities))

if col2.button('Calculate'):
    if selected_floor_no > selected_no_of_floor:
        col2.warning("Floor Number is higher than Total Number of Floor.")

    # One row, keyed by the SAME column names used in training,
    # so the order of features can never be wrong.
    row = {
        'Property Size': selected_property_size,
        'Property Age': selected_property_age,
        'No of Bathroom': selected_no_of_bathroom,
        'No of Cup Board': selected_no_of_cupboard,
        'Floor No ': selected_floor_no,
        'Total No of Floor': selected_no_of_floor,
        'Balconies': selected_no_of_balconies,
        'House Type': selected_house_type,
        'Lease Type': selected_lease_type,
        'Furnishing': selected_furniture,
        'Parking': selected_parking,
        'House Facing': selected_house_facing,
        'Water Supply': selected_water_supply,
        'Building Type': selected_building_type,
    }
    for label, column in amenities.items():
        row[column] = 1 if label in selected_amenities else 0

    user_input = pd.DataFrame([row])[saved['features']]
    rent = pipeline.predict(user_input)[0]
    col1.metric("Predicted Monthly Rent", f"₹ {rent:,.0f}")
