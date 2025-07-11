import subprocess
import threading
import time
import json
import pandas as pd
from datetime import datetime, timedelta
import requests
import tkinter as tk
from tkinter import messagebox
import pyttsx3
from live_graph_server import run_server, graph_updater

from algo import *
from oanda import *
from helper import *


def run_trading_script(credentials):
    """Stream live prices from OANDA and process them with the Algo instance,
    automatically reconnecting if the connection fails."""
    max_retries = 10  # Maximum number of reconnection attempts
    retry_count = 0
    retry_delay = 5  # Initial delay in seconds between retries
    transaction_types = []
    
    # Start transaction streaming in a separate thread with reconnection logic
    def run_transaction_stream():
        """Stream live transactions from OANDA with automatic reconnection."""
        transaction_max_retries = 10
        transaction_retry_count = 0
        transaction_retry_delay = 5
        
        while True:
            try:
                print(f"🔄 Starting/restarting OANDA transaction stream (attempt {transaction_retry_count + 1})")
                
                # Stream transactions with reconnection
                for transaction in stream_oanda_transactions(credentials):
                    # Transaction events are automatically handled in the stream function
                    # Reset retry count on successful data
                    transaction_retry_count = 0
                    transaction_retry_delay = 5

                    if transaction['type'] != 'HEARTBEAT':
                        print('trx======')
                        print(transaction['type'])
                        transaction_types.append(transaction['type'])
                    
                # If we exit the loop normally (generator ended), we should reconnect
                print("⚠️ Transaction stream ended. Attempting to reconnect...")
                transaction_retry_count += 1
                
            except Exception as e:
                # Handle any exceptions that might occur during transaction streaming
                transaction_retry_count += 1
                print(f"❌ Error in transaction stream: {e}")
                print(f"⏱️ Reconnecting transaction stream in {transaction_retry_delay} seconds...")
            
            # Check if we've exceeded max retries for transactions
            if transaction_max_retries > 0 and transaction_retry_count >= transaction_max_retries:
                print(f"❌ Transaction stream failed to connect after {transaction_max_retries} attempts. Giving up.")
                say_nonblocking("Transaction stream connection failed after multiple attempts.", voice="Victoria")
                break
                
            # Exponential backoff for transaction retry delay (up to 60 seconds)
            time.sleep(transaction_retry_delay)
            transaction_retry_delay = min(transaction_retry_delay * 1.5, 60)  # Increase delay, but cap at 60 seconds
    
    # Start transaction streaming thread
    transaction_thread = threading.Thread(target=run_transaction_stream, daemon=True)
    transaction_thread.start()
    
    while True:
        try:
            print(f"🔄 Starting/restarting OANDA price stream (attempt {retry_count + 1})")
            # Stream live prices from OANDA and process them with the Algo instance
            for price in stream_oanda_live_prices(credentials, instrument):
                take, return_dict = purple.process_row(price['timestamp'], price['bid'], precision, say=True)
                if take:
                    say_nonblocking(f'We would take a trade now! {take}.')

                # Get the latest transactions
                if transaction_types:
                    print('trx list=====')
                    print(transaction_types)
                    transaction_types.clear()

                # Update the live graph
                graph_updater.update_graph(return_dict)
                # Reset retry count on successful data
                retry_count = 0
                retry_delay = 5
                
            # If we exit the loop normally (generator ended), we should reconnect
            print("⚠️ Price stream ended. Attempting to reconnect...")
            retry_count += 1
            
        except Exception as e:
            # Handle any exceptions that might occur during streaming
            retry_count += 1
            print(f"❌ Error in trading script: {e}")
            print(f"⏱️ Reconnecting in {retry_delay} seconds...")
        
        # Check if we've exceeded max retries
        if max_retries > 0 and retry_count >= max_retries:
            print(f"❌ Failed to connect after {max_retries} attempts. Giving up.")
            say_nonblocking("Connection failed after multiple attempts. Please check your network and restart the application.", voice="Alex")
            break
            
        # Exponential backoff for retry delay (up to 60 seconds)
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 1.5, 60)  # Increase delay, but cap at 60 seconds


def toggle_take():
    global take
    take = not take
    take_button.config(text=f"Take: {'ON' if take else 'OFF'}")
    say_nonblocking(f"Take is now {'ON' if take else 'OFF'}")

take = False

# Load secrets from secrets.json
with open('secrets.json', 'r') as f:
    credentials = json.load(f)

instrument = input("Instrument (e.g., USD_CAD): ")

precision = get_instrument_precision(credentials, instrument)  # Get precision from the mean price
purple = Algo(base_interval='15min', slow_interval='30min',aspr_interval='3min', peak_interval='2min')  # Create an instance of the Algo class with 15-minute intervals


# Start the Flask server in a background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("🌐 Live graph server started at http://127.0.0.1:5000")
print("Open your web browser and navigate to the URL above to see the live graph")
print("📝 Controls: Spacebar = pause/resume updates, R = refresh now")

# Wait a moment for server to start
time.sleep(3)

rows = int(input("Number of rows to fetch for live data (default 5000): "))
rows = min(rows, 1000)  # Limit rows to 1000 to reduce startup time

# Before streaming, get the historical data for that instrument from oanda
historical_data = get_oanda_data(
    credentials=credentials,
    instrument=instrument,
    granularity='S5',  # 5-second granularity
    hours=8,  # Fetch 8 hours of historical data
    rows=rows  # Fetch up to 1000 rows of historical data
)

historical_df = pd.DataFrame(historical_data)
historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
historical_df = historical_df.drop_duplicates(subset=['timestamp'], keep='last')

# Process historical data in batches to reduce memory usage
batch_size = 100
for i in range(0, len(historical_df), batch_size):
    batch = historical_df.iloc[i:i + batch_size]
    for _, row in batch.iterrows():
        timestamp = row['timestamp']
        price = round(row['price'], precision)
        
        # Process the historical data row with the Algo instance
        take, return_dict = purple.process_row(timestamp, price, precision, say=False)
        
        # Update the live graph with historical data
        graph_updater.update_graph(return_dict)

# Start the trading script in a separate thread
trading_thread = threading.Thread(target=run_trading_script, args=(credentials,), daemon=True)
trading_thread.start()

# Run the Tkinter GUI in the main thread
root = tk.Tk()
root.title("Trading GUI")

# Add a button to the GUI
hello_button = tk.Button(root, text="Say Hello", command=say_hello)
hello_button.pack(pady=20)

# Add the toggle button for 'Take'
take_button = tk.Button(root, text="Take: OFF", command=toggle_take)
take_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()

