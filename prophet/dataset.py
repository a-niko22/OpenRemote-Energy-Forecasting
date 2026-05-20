import json
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()

# Input variables:
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
host_name = os.getenv("HOST_NAME")
realm_name = os.getenv("REALM_NAME")
asset_id = os.getenv("ASSET_ID")
attribute_name = os.getenv("ATTRIBUTE_NAME")

current_date = datetime.now().date()
start_date = current_date - timedelta(days=365)
end_date = current_date + timedelta(days=2)


def get_access_token(client_id, client_secret, token_url):
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    access_token = None

    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token_response = response.json()
        access_token = token_response['access_token']
    except Exception as e:
        print("Get access token failed:", e)

    return access_token


if __name__ == "__main__":
    token_url = "https://" + host_name + "/auth/realms/" + realm_name + "/protocol/openid-connect/token"
    access_token = get_access_token(client_id, client_secret, token_url)

    if access_token:
        api_command = f'/asset/datapoint/{asset_id}/{attribute_name}'
        api_url = f'https://{host_name}/api/{realm_name}{api_command}'
        print(api_url)

        payload = json.dumps({
            "fromTimestamp": 0,
            "toTimestamp": 0,
            "fromTime": f"{start_date}T00:00:00.000Z",
            "toTime": f"{end_date}T00:00:00.000Z",
            "type": "string"
        })

        print(str(payload))

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }

        response = requests.request("POST", api_url, headers=headers, data=payload)

        # print(response.text)
        if not response.ok:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:200]}")

        # Convert string to Python objects
        data = json.loads(response.text)

        # Convert milliseconds to datetime
        x_values = [datetime.fromtimestamp(point["x"] / 1000) for point in data]
        y_values = [point["y"] for point in data]

        # Plot
        plt.figure(figsize=(16, 9))
        plt.plot(x_values, y_values, marker='.')
        plt.axvline(x=mdates.date2num(datetime.now()), color='black', linestyle='--', label="current datetime")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Plot")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()