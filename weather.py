import requests

api_key = "87da512c927153e99b20d6a25c108aff"
city = input("Please enter a city: ")

url = f'https://api.openweathermap.org/data/2.5/weather?lq={city}&appid={api_key}'

data = requests.get(url).json()

temp = data['main']['temp']
humidity = data['main']['humidity']