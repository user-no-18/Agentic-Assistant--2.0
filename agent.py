import os
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


# Tool implementations

def news_search(query: str) -> str:
    """Search recent news articles for a topic."""
    tavily = TavilySearchResults(
        k=3,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        search_kwargs={"topic": "news", "search_depth": "basic", "include_answer": True},
    )
    return str(tavily.invoke({"query": query}))


async def weather(location: str) -> str:
    """Get current weather for a city name."""
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_data = geo_response.json()
        if not geo_data.get("results"):
            return f'Location "{location}" not found'

        place = geo_data["results"][0]
        weather_response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            },
        )
        current = weather_response.json()["current"]
        return (
            f'{place["name"]}, {place.get("country", "")}: '
            f'{current["temperature_2m"]}°C, Wind: {current["wind_speed_10m"]} km/h, '
            f'Humidity: {current["relative_humidity_2m"]}%'
        )


async def currency_conversion(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert one currency into another."""
    import httpx

    source = from_currency.upper()
    target = to_currency.upper()
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(f"https://api.exchangerate-api.com/v4/latest/{source}")
        data = response.json()
        rate = data.get("rates", {}).get(target)
        if not rate:
            return f"Currency conversion to {target} not found"
        return f"{amount} {source} = {amount * rate:.2f} {target}"


async def country_info(country: str) -> str:
    """Get capital, population, region, and languages for a country."""
    import httpx

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            f"https://restcountries.com/v3.1/name/{country}",
            params={"fullText": "false"},
        )
        data = response.json()
        if isinstance(data, dict) and data.get("status") == 404:
            return f'Country "{country}" not found'
        if not isinstance(data, list) or not data:
            return f'Country "{country}" not found'

        info = data[0]
        capital = (info.get("capital") or ["N/A"])[0]
        population = f'{info.get("population", 0):,}'
        languages = ", ".join(info.get("languages", {}).values()) or "N/A"
        return (
            f'{info["name"]["common"]}: Capital: {capital}, '
            f'Population: {population}, Region: {info.get("region", "N/A")}, '
            f'Languages: {languages}'
        )


def build_agent() -> Any:
    tavily = TavilySearchResults(
        k=3,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        search_kwargs={"topic": "general", "search_depth": "basic", "include_answer": True},
    )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000))

    tools = [
        tavily,
        StructuredTool.from_function(
            func=wikipedia.invoke,
            name="wikipedia",
            description="Search Wikipedia for factual, historical, or general knowledge questions.",
        ),
        StructuredTool.from_function(
            func=news_search,
            name="news_search",
            description="Search recent news articles for a topic.",
        ),
        StructuredTool.from_function(
            coroutine=weather,
            name="weather",
            description="Get current weather for a city name.",
        ),
        StructuredTool.from_function(
            coroutine=currency_conversion,
            name="currency_conversion",
            description="Convert an amount between currency codes such as USD and EUR.",
        ),
        StructuredTool.from_function(
            coroutine=country_info,
            name="country_info",
            description="Get information about a country's capital, population, region, and languages.",
        ),
    ]

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return create_react_agent(model, tools, checkpointer=MemorySaver())
