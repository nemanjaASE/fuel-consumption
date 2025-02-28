from flask import Flask, request, render_template, send_file
from helpers import load
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import os

model, features, scaler, kmeans = load.load_models()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/vehicle', methods=['GET', 'POST'])
def vehicle():
    if request.method == 'POST':
        try:

            new_data = pd.DataFrame([{
                "Fuel Consumption City (L/100 km)": float(request.form['city']),
                "Fuel Consumption Comb (L/100 km)": float(request.form['comb']),
                "Fuel Consumption Hwy (L/100 km)": float(request.form['hwy']),
                "Engine Size(L)": float(request.form['engine_size']),
                "Cylinders": int(request.form['cylinders']),
                "Fuel Type": request.form['fuel_type']
            }])

            new_data = pd.get_dummies(new_data, columns=["Fuel Type"], drop_first=True)

            for col in features:
                if col not in new_data.columns:
                    new_data[col] = 0  

            new_data = new_data[features]  
            print(model)
            prediction = model.predict(new_data)

            return render_template('co2_result.html', co2_emission=prediction[0])

        except Exception as e:
            print(e)
            return render_template('vehicle.html'), 400
    return render_template('vehicle.html')

@app.route('/brand', methods=['GET', 'POST'])
def brand():
    df = pd.read_csv('./data/fuel_consumption.csv')
    brands = load.load_unique_brands()

    if request.method == 'POST':
        brand = request.form['brand']
        
        brand_data = df[df['Make'] == brand]
        features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']
        feature_contributions = {}

        for feature in features:
            feature_contributions[feature] = brand_data[feature].mean()

        labels = ['ENG. SIZE', 'CYLINDERS', 'CITY FUEL', 'HWY FUEL', 'COMB. FUEL']
        sizes = feature_contributions.values()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=10, colors=plt.cm.Paired.colors)
        ax.set_title(f'CO2 Emission Factors for {brand} Brand')

        img_path = os.path.join('static', 'images', f'{brand}_co2_chart.png')
        fig.savefig(img_path)
        plt.close(fig)

        return render_template('brand.html', brands=brands, image_url=f'/static/images/{brand}_co2_chart.png')

    brands = df['Make'].unique()
    return render_template('brand.html', brands=brands)

@app.route('/economy', methods=['GET', 'POST'])
def economy():
    if request.method == 'POST':
        try:

            new_data = pd.DataFrame([{
                "Fuel Consumption City (L/100 km)": float(request.form['city']),
                "Fuel Consumption Comb (L/100 km)": float(request.form['comb']),
                "Fuel Consumption Hwy (L/100 km)": float(request.form['hwy']),
                "Engine Size(L)": float(request.form['engine_size']),
                "Cylinders": int(request.form['cylinders']),
                "CO2 Emissions(g/km)": request.form['emissions']
            }])

            new_vehicle_scaled = scaler.transform(new_data)

            cluster = kmeans.predict(new_vehicle_scaled)

            if cluster == 0:
                category = 'Economic vehicle'
            elif cluster == 1:
                category = 'Medium economic vehicle'
            else:
                category = 'Non-economic vehicle'

            return render_template('economy_result.html', category=category)

        except Exception as e:
            return render_template('economy.html'), 400
    return render_template('economy.html')

@app.route('/co2-bar-chart')
def co2_bar_chart():
    df = pd.read_csv('./data/fuel_consumption.csv')

    brand_emissions = df.groupby('Make')['CO2 Emissions(g/km)'].mean().reset_index()
    brand_emissions_sorted = brand_emissions.sort_values(by='CO2 Emissions(g/km)', ascending=False)

    labels = brand_emissions_sorted['Make']
    sizes = brand_emissions_sorted['CO2 Emissions(g/km)']

    fig, ax = plt.subplots(figsize=(14, 12))
    bar_width = 0.6
    ax.bar(labels, sizes, width=bar_width, color='skyblue')

    ax.set_xlabel('Car Brand', fontsize=16)
    ax.set_ylabel('Average CO2 Emission (g/km)', fontsize=14)
    ax.set_title('Average CO2 Emission per Car Brand', fontsize=14)
    ax.tick_params(axis='x', rotation=80, labelsize=12, pad=10) 
    ax.tick_params(axis='y', labelsize=12)

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)