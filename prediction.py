import requests
url = 'http://localhost:5000/predict'
data = {
     #"features": [22.0, 1, 71948.0, 0, 2, 35000.0, 3, 16.02, 0.49, 3.0, 561, 0, 1],
     "features": [40.0, 1, 123456.0, 0, 2, 65000.0, 3, 10.02, 0.39, 3.0, 661, 0, 1]

}
response=requests.post(url,json=data)
print(response.json())