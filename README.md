# Predict-location
다음으로 이동할 이동 경로를 예측합니다.

# LSTM.py
(1) make_model : Creating a LSTM model
(2) pred_route : Predict the next location to move to (using LSTM model)

# make_map.py
(1) create_df 
- Using the output made of LSTM.pred_route
- Creating a dataframe for creating a map

(2) create_map
- Using the output made of make_map.create_df
- Creating a map

# Data
CDR
- Preprocessing CDR data
- Combining path of location each of p_id
