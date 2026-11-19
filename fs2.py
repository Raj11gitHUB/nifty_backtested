import os
import configparser
import webbrowser
from fyers_apiv3 import fyersModel

# === Config loading ===
config = configparser.ConfigParser()
ini_path = os.path.join(os.path.dirname(__file__), 'm1.ini')
config.read(ini_path)

client_id = config["fyers"]["client_id"]
secret_key = config["fyers"]["secret_key"]
redirect_uri = config["fyers"]["redirect_uri"]

# File to store access token
token_file = os.path.join(os.path.dirname(__file__), 'Ts.txt')

# === Utility functions ===
def read_token():
    if not os.path.exists(token_file):
        return None
    with open(token_file, "r") as f:
        return f.read().strip()

def write_token(token):
    with open(token_file, 'w') as f:
        f.write(token)

def is_token_valid(token):
    try:
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=token, log_path="")
        res = fyers.get_profile()
        return res.get("s") == "ok"
    except Exception as e:
        print("Token validation failed:", e)
        return False

# === Authentication Flow ===
def get_auth_code():
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code"
    )
    url = session.generate_authcode()
    print(f"Open the following URL in your browser:\n{url}")
    webbrowser.open(url)
    return input("Paste the auth code from the URL here: ").strip()

def get_access_token(auth_code):
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )
    session.set_token(auth_code)
    response = session.generate_token()
    token = response["access_token"]
    write_token(token)
    print("Access token obtained and saved.")
    return token

# === API Action ===
def get_holdings(token):
    fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=token, log_path="")
    response = fyers.holdings()
    print("Holdings:", response)

# === Main flow ===
def main():
    token = read_token()
    if not token or not is_token_valid(token):
        print("No valid token found. Starting authentication flow...")
        auth_code = get_auth_code()
        token = get_access_token(auth_code)
    get_holdings(token)

if __name__ == "__main__":
    main()
