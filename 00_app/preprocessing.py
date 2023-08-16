import numpy as np
import pandas as pd
import pickle

with open('lr_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

intercept = model['intercept']
coefficients = model['coefficients']

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg',
'popularity']

def data_pipeline(json_data):
    df = pd.DataFrame([json_data])
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        value = (df['number_of_doors'] == v).astype(int)
        df[feature] = value
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',               'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:              
        feature = 'is_type_%s' % v        
        df[feature] = (df['engine_fuel_type'] == v).astype(int)        
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive',              'all_wheel_drive', 'four_wheel_drive']:            
        feature = 'is_driven_wheels_%s' % v        
        df[feature] = (df['driven_wheels'] == v).astype(int)        
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:    
        feature = 'is_mc_%s' % v        
        df[feature] = (df['market_category'] == v).astype(int)        
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:            
        feature = 'is_size_%s' % v        
        df[feature] = (df['vehicle_size'] == v).astype(int)        
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible',              '4dr_hatchback']:          
        feature = 'is_style_%s' % v        
        df[feature] = (df['vehicle_style'] == v).astype(int)        
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    try:

        # Calculate the predicted price using the loaded model
        predicted_price = intercept + X.dot(coefficients)
        predicted_price = predicted_price[0]

        return predicted_price.tolist()  # Convert to a list for JSON serialization
    except Exception as e:
        return {'error': str(e)}

