import gradio as gr
import pandas as pd
import numpy as np
from prophet import Prophet
import random
df = pd.read_csv('df60')
place_df = pd.read_csv('place.csv')
Station = pd.DataFrame(np.load('Station.npy',allow_pickle=True))
def findProvince(Province):
    if Province == 'Bangkok':
        name = list(Station[Station[6]=='กรุงเทพมหานคร'][0])[0]
    return name

def get_weather_forecast(Province,Activity,Purpose,Year,Month,Date):
    name = findProvince(Province)
    p_df = df.rename(columns={'Date_time':'ds',str(name):'y'})
    rain_df =  p_df[p_df['mode']=='Rain'][['ds','y']]
    rain_model = Prophet(interval_width=0.95)
    rain_model.fit(rain_df)
    date_time = pd.Series(pd.to_datetime(f'{Year}-{Month}-{Date}',format = '%Y-%m-%d'),name='ds')
    date_time = pd.DataFrame(date_time)
    forecast = rain_model.predict(date_time)
    
    
    if float(forecast['yhat'][0]) >= 5:
        WeatherCon = 'SUNNY'
        txt = 'น้อย'
    else:
        WeatherCon = 'RAIN'
        txt = 'สูง'
    if Activity == 'Indoor':
        IsIndoor = 'Y'
    else:
        IsIndoor = 'N'
        
    place = place_df[place_df['Purpose']==Purpose]
    place = place[place['IsIndoor']==IsIndoor]
    place = place[place['Province']==Province]
    if WeatherCon == 'SUNNY':
        place = place[place['WeatherCon']==WeatherCon]
    place = place['LocationName']
    #random = random.randrange(len(place))
    place = list(place)[0]
    return f'มีโอกาสฝนตก{txt} สถานที่ที่แนะนำคือ:{place}'
    

iface = gr.Interface(
    fn = get_weather_forecast,
    inputs = [
        
        gr.Dropdown(
            ["Bangkok"], label="Province", info="Will add more later!"
        ),

        gr.Dropdown(
            ["Indoor","Outdoor"], label="Activity"
        ),

        gr.Dropdown(
            ["Entertainment","Culture",'Shop','Science'], label="Purpose"
        ),

        gr.Dropdown(
            [2023], label="Year", info="Will add more later!"
        ),

        gr.Dropdown(
            [1,2,3,4,5,6,7,8,9,10,11,12], label="Month"
        ),

        gr.Dropdown(
            [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], label="Date"
        )

    ],
    
    outputs=gr.outputs.Textbox(label="Prediction"),
    live=True,
    title="Weather Forecast",
    description="Get the weather forecast for a city.",
    theme="default",
)

iface.launch()
