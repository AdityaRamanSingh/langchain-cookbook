from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    FinalAnswerTool,
    load_tool,
    tool,
)

from smolagents.gradio_ui import GradioUI

import datetime
import requests
import pytz
import yaml


@tool
def ny_custom_tool(arg1: str, arg2: str) -> str:
    """A custom tool that does nothing yet.

    Args:
        arg1: First argument.
        arg2: Second argument.
    """
    return "What magic will you build?"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Fetches current local time for a specified timezone.

    Args:
        timezone: A string representing a valid timezone
        (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        return f"The current local time in {timezone} is: {local_time}"

    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Fetches current weather for a city.

    Args:
        city: Name of the city, e.g. 'Indore'.
    """
    key = os.environ["OPENWEATHER_API_KEY"]
    url = "https://api.openweathermap.org/data/2.5/weather"
    r = requests.get(url, params={"q": city, "appid": key, "units": "metric"})
    d = r.json()
    if d.get("cod") != 200:
        return f"Could not find weather for {city}."
    return (
        f"{city}: {d['main']['temp']}°C, "
        f"{d['weather'][0]['description']}, "
        f"humidity {d['main']['humidity']}%"
    )

# Final answer tool
final_answer = FinalAnswerTool()


# Model
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)


# Image generation tool
image_generation_tool = load_tool(
    "agents-course/text-to-image",
    trust_remote_code=True,
)


# Load prompts
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)


# Create coding agent
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        image_generation_tool,
    ],
    max_steps=6,
    verbosity_level=1,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
)


# Launch Gradio UI
GradioUI(agent).launch()
