import pandas as pd
import numpy as np
import plotly.express as px
from tqdm.notebook import tqdm
import os
import folium

def create_df(data, route, save_csv = None):
    
    if type(route) != list :
        print('route의 type은 list입니다.')
    else:
        df_new = pd.DataFrame([], columns = ['address', 'new_bts_id', 'lat', 'long'])
        temp2 = pd.DataFrame([], columns = ['address', 'new_bts_id', 'lat', 'long'])

        for i in tqdm(range(len(route))):
            contain_a = data['new_bts_id'].str.contains(route[i])
            contain_b = data[contain_a]

            address = contain_b['address'].tolist()[0]
            bts_id = contain_b['new_bts_id'].tolist()[0]
            point_lat = contain_b['lat'].tolist()[0]
            point_long = contain_b['long'].tolist()[0]

            for j in range(1):        
                temp = pd.DataFrame([[address, bts_id, point_lat, point_long]],
                                   columns = ['address', 'new_bts_id', 'lat', 'long'])

                temp2 = pd.concat([temp2, temp], axis = 0)

            target_data = pd.concat([df_new, temp2], axis = 0).reset_index(drop = True)
            
    if save_csv == True :
        target_data.to_csv('target_data.csv',mode = 'w')
        
    return target_data

def create_map(data, map_name, zoom_start=13, line_opacity = 0.5):
    my_map = folium.Map(location = [data['lat'][0], data['long'][0]],  zoom_start = zoom_start ) 
    
    # CircleMarker with radius 
    for i in range(len(data['lat'])):
        folium.Marker([data['lat'][i], data['long'][i]], popup = data['new_bts_id'][i]).add_to(my_map)
        
    lat_long = []
    for i in range(data.shape[0]):
        lat_long.append((data['lat'][i], data['long'][i]))
        
    folium.PolyLine(locations = lat_long,
                    line_opacity = line_opacity).add_to(my_map) 
    
    if type(map_name) == str:
        save_name = map_name + '.html'
        my_map.save(save_name) 
        print('지도 그림을 저장했습니다.')
    else :
        print('map_name의 type은 str이어야 합니다.', '\n' 'map_name을 map.html로 저장합니다.')
        my_map.save("map.html") 