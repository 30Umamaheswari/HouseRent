import streamlit as st
# import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="House_Rent Calculator",
                   layout="wide", )

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

# with open(r'house_model.pkl', 'rb') as f_h:
#     loaded_model = pickle.load(f_h)
#
# with open('house_facing_label.pkl', 'rb') as file_hf:
#     loaded_house_facing = pickle.load(file_hf)
#
# with open('house_type_label.pkl', 'rb') as file_ht:
#     loaded_house_type = pickle.load(file_ht)
#
# with open('lease_type_label.pkl', 'rb') as file_lt:
#     loaded_lease_type = pickle.load(file_lt)
#
# with open('furnishing_label.pkl', 'rb') as file_f:
#     loaded_Furnishing = pickle.load(file_f)
#
# with open('Parking_label.pkl', 'rb') as file_p:
#     loaded_Parking = pickle.load(file_p)
#
# with open('Water_Supply_label.pkl', 'rb') as file_ws:
#     loaded_Water_Supply = pickle.load(file_ws)
#
# with open('Building_Type_label.pkl', 'rb') as file_ws:
#     loaded_Building_Type = pickle.load(file_ws)


with open('house_model_and_encoders.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Extract loaded model and label encoders
loaded_house_model = loaded_data['model']
loaded_house_facing_label_encoder = loaded_data['house_facing_label_encoder']
loaded_house_type_label_encoder = loaded_data['house_type_label_encoder']
loaded_lease_type_label_encoder = loaded_data['lease_type_label_encoder']
loaded_furnishing_label_encoder = loaded_data['furnishing_label_encoder']
loaded_parking_label_encoder = loaded_data['parking_label_encoder']
loaded_water_supply_label_encoder = loaded_data['water_supply_label_encoder']
loaded_building_type_label_encoder = loaded_data['building_type_label_encoder']

# creating list
house_type = ['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS']

lease_type = ['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY']

furniture = ['Semi Furnished', 'Fully Furnished', 'Not Furnished']

parking = ['Both', 'Two Wheeler', 'No Parking', 'Four Wheeler']

house_facing = ['NE', 'E', 'S', 'N', 'SE', 'W', 'NW', 'SW']

water_supply = ['Corporation', 'Corporation Bore', 'BoreWell']

building_type = ['Apartment', 'Independent House', 'Independent Floor', 'Guest House']

# Streamlit app Creation
col1, col2, col3 = st.columns([8, 4, 4], gap='large')

selected_house_type = col2.selectbox("Select House Type", house_type, key=1)
selected_lease_type = col2.selectbox("Select Lease Type", lease_type, key=2)
selected_furniture = col2.selectbox("Select type of Furnishing", furniture, key=3)
selected_parking = col2.selectbox("Select type of Parking", parking, key=4)
selected_house_facing = col2.selectbox("Choose House Facing", house_facing, key=5)
selected_water_supply = col2.selectbox("Choose Water Supply", water_supply, key=6)
selected_building_type = col2.selectbox("Building Type", building_type, key=7)

# Encoding

house_type_label = loaded_house_type_label_encoder.transform([selected_house_type])[0]
lease_type_label = loaded_lease_type_label_encoder.transform([selected_lease_type])[0]
furniture_label = loaded_furnishing_label_encoder.transform([selected_furniture])[0]
parking_label = loaded_parking_label_encoder.transform([selected_parking])[0]
house_facing_label = loaded_house_facing_label_encoder.transform([selected_house_facing])[0]
water_supply_label = loaded_water_supply_label_encoder.transform([selected_water_supply])[0]
building_type_label = loaded_building_type_label_encoder.transform([selected_building_type])[0]

# house features
selected_property_size = col3.number_input("Property Size", min_value=1, max_value=13000)
selected_property_age = col3.number_input("Property Age", min_value=1, max_value=100)
selected_no_of_bathroom = col3.number_input("Number of Bathroom", min_value=1, max_value=6)
selected_no_of_cupboard = col3.number_input("Number of Cub Board", min_value=0, max_value=15)
selected_floor_no = col3.number_input("Floor Number", min_value=0, max_value=20)
selected_no_of_floor = col3.number_input("Total Number of Floor", min_value=0, max_value=20)
selected_no_of_balconies = col3.number_input("Total Number of Balcony", min_value=0, max_value=6)

amenities = ['Negotiable', 'Lift', 'Gym', 'Internet', 'Air Conditioning', 'Club', 'Intercom', 'Swimming Pool', 'Car Parking',
             'Servant', 'Security', 'Shopping Center', 'Gas Pipeline', 'Park', 'Rain Water Harvesting',
             'Sewage Treatment Plant' 'House Keeping', 'Power Backup', 'Visitor Parking']

selected_amenities = col3.multiselect("Select amenities", amenities)


if col2.button('Calculate'):
    user_input = [selected_property_size, selected_property_age, selected_no_of_bathroom, selected_no_of_cupboard,
                  selected_floor_no, selected_no_of_floor, selected_no_of_balconies]

    # Add amenities features
    user_input.extend([1 if amenity in selected_amenities else 0 for amenity in amenities])

    # Add label-encoded categorical features
    user_input.extend([house_facing_label, house_type_label, lease_type_label,
                       furniture_label, parking_label, water_supply_label, building_type_label])

    missing_feature_value = 0  # You can adjust this based on your data

    # Add a placeholder for the missing feature
    user_input.append(missing_feature_value)

    # Reshape user_input into a 2D array
    user_input_2d = [user_input]

    rent = loaded_house_model.predict(user_input_2d)
    # x = round(float(rent[0]))
    st.write("Predicted Rent:", rent)
