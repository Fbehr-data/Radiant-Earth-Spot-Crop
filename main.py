import fastapi 
import uvicorn
#import movie_data
#import weather
import socket
import requests

#requests.get('https://{}:8000'.format(socket.gethostbyname(socket.gethostname()))) 

app = fastapi.FastAPI()

@app.get('/')
def index():
    return {
        "message": "Hello World"
    }

@app.get('/api/movie/{title}')
async def movie_search(title: str):
    movie = await movie_data.get_movie()
    raise fastapi.HTTPException(status_code=404)
    raise fastapi.HTTPException(status_code=400)

if __name__ == '__main__':
    uvicorn.run(app)