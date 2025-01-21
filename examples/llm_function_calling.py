from openai import OpenAI
import json
import requests
from typing import Union
import os


OMNISTACK_API_KEY=os.environ.get('OMNISTACK_API_KEY')

client = OpenAI(base_url="https://api.omnistack.sh/openai/v1", api_key=OMNISTACK_API_KEY)

# This is your OmniStack Model ID, which could be obtained here https://console.omnistack.sh/
# You can OpenAI models, LLAMA or any other models that support function calling. 
model= "brianne_enoch_victoria" 
system_prompt = "You are a helpful assistant. if no data available, return N/A."

# Define functions 
functions = [
    {
        "name": "get_stock_price",
        "description": "Fetch the current stock price of a specific publicly traded company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., IBM, TSLA) for a publicly traded company, to fetch current price."
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_order_book",
        "description": "Fetch the order book data for a specific trading pair or stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., IBM, TSLA) to fetch order book."
                }
            },
            "required": ["ticker"]
        }
    }
]

def get_stock_price(ticker: str) -> dict:
    # Using Alpha Vantage demo API to fetch stock prices. With demo API key it works only for IBM stock. 
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": "5min",
        "apikey": "demo"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        time_series = data.get("Time Series (5min)", {})

        # Extract the latest price based on the most recent timestamp
        if time_series:
            latest_timestamp = sorted(time_series.keys())[-1]
            latest_data = time_series[latest_timestamp]
            return {
                "ticker": ticker,
                "price": latest_data.get("1. open", "N/A")
            }

    return {
        "ticker": ticker,
        "price": f"Price not available for {ticker} company."
    }

def process_model_response(model_response) -> Union[dict, None]:
    message = model_response.choices[0].message

    # Check if a function call exists and is valid
    if not (hasattr(message, "function_call") and message.function_call):
        return None

    tool_name = message.function_call.name
    tool_args = message.function_call.arguments

    # Ensure tool name and arguments are valid
    if not (tool_name and tool_args):
        print(f"Error: Invalid tool name or arguments: {tool_name}, {tool_args}")
        return None

    try:
        tool_args = json.loads(tool_args)
    except json.JSONDecodeError:
        print("Error: Invalid JSON arguments from model")
        return None

    # Handle the tool call
    if tool_name == "get_stock_price":
        return get_stock_price(**tool_args)
    
    print(f"Error: Unknown tool '{tool_name}' called by the model")
    return None

def chat_with_model(user_input: str) -> None:
    try:
        # Initial model call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            functions=functions,
            function_call="auto"
        )
        
        # Process model response and invoke functions if requested
        function_result = process_model_response(response)
        
        if not function_result:
            print("No function call needed")
            print(f"Response: {response.choices[0].message.content}")
            return

        # Send the function's output back to the model
        follow_up_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "null", "function_call": response.choices[0].message.function_call},
                {"role": "function", "name": response.choices[0].message.function_call.name, "content": json.dumps(function_result)}
            ],
            functions=functions,
            function_call="auto"
        )
        
        print(f"Initial Function Call Request: {response.choices[0].message.function_call}")
        print(f"Final response: {follow_up_response.choices[0].message.content}")
    
    except Exception as e:
        print(f"Error during chat interaction: {str(e)}")

# Example usage
# It will return stock price only for IBM, as it's the only supported symbol on https://www.alphavantage.co/ API with a demo API key. 
user_input = "What's the IBM stock price today?"
#user_input = "What's the Apple stock price today?"
#user_input = "Tell me a joke"
chat_with_model(user_input)