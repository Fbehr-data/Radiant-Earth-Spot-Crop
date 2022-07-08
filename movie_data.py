import httpx

async def get_movie(title_subtext: str):
    url = f'https://movieservice.talkpython.fm/api/search/{title_subtext}'
    
    async with httpx.AsyncClient() as client:
        resp: httpx.Response = await client.get(url)
        resp.raise_for_status()
        
        data =resp.json()
        
        print(resp, resp.text)
        
    results = data['hits']
    if not results:
        return None
    
    movie = results[0]
    return movie

