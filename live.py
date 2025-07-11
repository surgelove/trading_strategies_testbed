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



def stream_oanda_live_prices(credentials, instrument='USD_CAD', callback=None, max_duration=None):
    """
    Stream live prices from OANDA API for a single instrument
    
    Parameters:
    -----------
    credentials : dict
        Dictionary containing 'api_key' and 'account_id'
    instrument : str, default='USD_CAD'
        Single instrument to stream (e.g., 'EUR_USD', 'GBP_JPY')
    callback : function, optional
        Function to call with each price update: callback(timestamp, instrument, bid, ask, price)
    max_duration : int, optional
        Maximum streaming duration in seconds (None = unlimited)
    
    Returns:
    --------
    generator or None
        Yields price dictionaries or None if connection fails
    """
    
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')
    
    # LIVE OANDA STREAMING API URL
    STREAM_URL = "https://stream-fxtrade.oanda.com"
    
    print("🔴 CONNECTING TO OANDA LIVE STREAMING")
    print("=" * 50)
    print("⚠️  WARNING: This connects to LIVE market stream")
    print(f"📊 Streaming: {instrument}")
    print(f"⏱️  Duration: {'Unlimited' if max_duration is None else f'{max_duration}s'}")
    print("=" * 50)
    
    # Validate inputs
    if not api_key:
        print("❌ ERROR: Live API key is required!")
        return None
    
    if not account_id:
        print("❌ ERROR: Live Account ID is required!")
        return None
    
    # Headers for streaming request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/stream+json',
        'Content-Type': 'application/json'
    }
    
    # Streaming endpoint for prices - FIXED URL CONSTRUCTION
    stream_url = f"{STREAM_URL}/v3/accounts/{account_id}/pricing/stream"
    
    # Parameters for streaming
    params = {
        'instruments': instrument,
        'snapshot': 'true'  # Include initial snapshot
    }
    
    try:
        print(f"🌐 Initiating streaming connection...")
        print(f"   URL: {stream_url}")
        print(f"   Instrument: {instrument}")
        
        # Get instrument precision - FIXED: Use the BASE API URL, not streaming URL
        BASE_API_URL = "https://api-fxtrade.oanda.com"
        precision = get_instrument_precision(credentials, instrument)
        if precision is None:
            precision = 5  # Default precision
            print(f"⚠️  Using default precision: {precision}")

        # Make streaming request
        response = requests.get(stream_url, headers=headers, params=params, stream=True, timeout=30)
        
        # Check for HTTP errors
        if response.status_code == 401:
            print("❌ AUTHENTICATION ERROR (401)")
            print("   • Check your API key is correct")
            print("   • Ensure your API key has streaming permissions")
            return None
        elif response.status_code == 403:
            print("❌ FORBIDDEN ERROR (403)")
            print("   • Your account may not have streaming access")
            print("   • Check if your account is verified and funded")
            return None
        elif response.status_code == 404:
            print(f"❌ NOT FOUND ERROR (404)")
            print(f"   • Check instrument name: {instrument}")
            print(f"   • URL used: {stream_url}")
            return None
        elif response.status_code != 200:
            print(f"❌ HTTP ERROR {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        print("✅ Streaming connection established!")
        print("📈 Receiving live price updates...")
        print("   Press Ctrl+C to stop streaming")
        print("-" * 50)
        
        start_time = time.time()
        price_count = 0
        previous_price = None
        
        # Process streaming data line by line
        for line in response.iter_lines():
            # Check duration limit
            if max_duration and (time.time() - start_time) > max_duration:
                print(f"\n⏰ Reached maximum duration of {max_duration} seconds")
                break
                
            if line:
                try:
                    # Parse JSON data
                    data = json.loads(line.decode('utf-8'))
                    
                    # Handle different types of messages
                    if data.get('type') == 'PRICE':
                        timestamp = datetime.now()
                        
                        # Extract price information
                        instrument_name = data.get('instrument', instrument)
                        
                        # Get bid/ask prices
                        bids = data.get('bids', [])
                        asks = data.get('asks', [])
                        
                        if bids and asks:
                            bid_price = round(float(bids[0]['price']), precision)
                            ask_price = round(float(asks[0]['price']), precision)   
                            mid_price = round((bid_price + ask_price) / 2, precision)
                            spread_pips = (ask_price - bid_price) * 10000  # For most pairs
                            
                            # Skip if price hasn't changed
                            if previous_price is not None and bid_price == previous_price:
                                continue
                            
                            # Update previous price
                            previous_price = bid_price
                            price_count += 1
                            
                            # Create price dictionary
                            price_data = {
                                'timestamp': timestamp,
                                'instrument': instrument_name,
                                'bid': bid_price,
                                'ask': ask_price,
                                'price': mid_price,  # Mid price for compatibility
                                'spread_pips': round(spread_pips, 1),
                                'time': data.get('time', timestamp.isoformat()),
                                'tradeable': data.get('tradeable', True)
                            }
                            
                            # # Print price update (every 10th update to avoid spam)
                            # if price_count % 10 == 0:
                            #     print(f"💰 {timestamp.strftime('%H:%M:%S')} | {instrument_name} | "
                            #           f"Bid: {bid_price:.5f} | Ask: {ask_price:.5f} | "
                            #           f"Mid: {mid_price:.5f} | Spread: {spread_pips:.1f} pips")
                            
                            # Call callback function if provided
                            if callback:
                                try:
                                    callback(timestamp, instrument_name, bid_price, ask_price, mid_price)
                                except Exception as e:
                                    print(f"⚠️  Callback error: {e}")
                            
                            # Yield price data for generator usage
                            yield price_data
                            
                    elif data.get('type') == 'HEARTBEAT':
                        # Heartbeat to keep connection alive
                        if price_count % 100 == 0:  # Print occasionally
                            print(f"💓 Heartbeat - Connection alive ({price_count} prices received)")
                    
                    else:
                        # Other message types
                        print(f"📨 Message: {data.get('type', 'Unknown')} - {data}")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️  Processing error: {e}")
                    continue
        
        print(f"\n✅ Streaming completed. Total prices received: {price_count}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Streaming stopped by user. Total prices received: {price_count}")
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT ERROR: Streaming request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Lost connection to OANDA")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    finally:
        if 'response' in locals():
            response.close()


def stream_oanda_transactions(credentials, callback=None, max_duration=None):
    """
    Stream live transaction events from OANDA API (trades, orders, etc.)
    
    Parameters:
    -----------
    credentials : dict
        Dictionary containing 'api_key' and 'account_id'
    callback : function, optional
        Function to call with each transaction: callback(transaction_data)
    max_duration : int, optional
        Maximum streaming duration in seconds (None = unlimited)
    
    Returns:
    --------
    generator
        Yields transaction dictionaries
    """
    
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')
    
    # OANDA STREAMING API URL
    STREAM_URL = "https://stream-fxtrade.oanda.com"
    
    print("🔴 CONNECTING TO OANDA TRANSACTION STREAM")
    print("=" * 50)
    print("📊 Listening for: Orders, Trades, Fills, Modifications")
    print("=" * 50)
    
    # Validate inputs
    if not api_key or not account_id:
        print("❌ ERROR: API key and Account ID are required!")
        return None
    
    # Headers for streaming request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/stream+json',
        'Content-Type': 'application/json'
    }
    
    # Streaming endpoint for transactions
    stream_url = f"{STREAM_URL}/v3/accounts/{account_id}/transactions/stream"
    
    try:
        print(f"🌐 Initiating transaction stream connection...")
        
        # Make streaming request
        response = requests.get(stream_url, headers=headers, stream=True, timeout=30)
        
        # Check for HTTP errors
        if response.status_code != 200:
            print(f"❌ HTTP ERROR {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        print("✅ Transaction stream connection established!")
        print("📈 Listening for transaction events...")
        print("   Press Ctrl+C to stop streaming")
        print("-" * 50)
        
        start_time = time.time()
        transaction_count = 0
        
        # Process streaming data line by line
        for line in response.iter_lines():
            # Check duration limit
            if max_duration and (time.time() - start_time) > max_duration:
                print(f"\n⏰ Reached maximum duration of {max_duration} seconds")
                break
                
            if line:
                try:
                    # Parse JSON data
                    data = json.loads(line.decode('utf-8'))
                    
                    # Handle different types of transaction messages
                    transaction_type = data.get('type')
                    
                    if transaction_type == 'ORDER_FILL':
                        # Trade execution
                        instrument = data.get('instrument')
                        units = data.get('units')
                        price = data.get('price')
                        time_stamp = data.get('time')
                        trade_id = data.get('tradeOpened', {}).get('tradeID')
                        
                        side = "BUY" if float(units) > 0 else "SELL"
                        
                        print(f"🎯 TRADE FILLED: {side} {abs(float(units))} {instrument} @ {price}")
                        print(f"   Trade ID: {trade_id}")
                        print(f"   Time: {time_stamp}")
                        
                        # Speak the trade execution
                        if side == "BUY":
                            say_nonblocking(f"Trade executed! Bought {instrument} at {price}", voice="Alex")
                        else:
                            say_nonblocking(f"Trade executed! Sold {instrument} at {price}", voice="Samantha")
                        
                    elif transaction_type == 'ORDER_CANCEL':
                        # Order cancellation
                        order_id = data.get('orderID')
                        reason = data.get('reason')
                        print(f"❌ ORDER CANCELLED: Order ID {order_id} - Reason: {reason}")
                        
                    elif transaction_type == 'STOP_LOSS_FILL':
                        # Stop loss triggered
                        instrument = data.get('instrument')
                        units = data.get('units')
                        price = data.get('price')
                        trade_id = data.get('tradeID')
                        
                        print(f"🛑 STOP LOSS TRIGGERED: {instrument} @ {price}")
                        print(f"   Trade ID: {trade_id}")
                        say_nonblocking(f"Stop loss triggered on {instrument} at {price}", voice="Victoria")
                        
                    elif transaction_type == 'TAKE_PROFIT_FILL':
                        # Take profit triggered
                        instrument = data.get('instrument')
                        units = data.get('units')
                        price = data.get('price')
                        trade_id = data.get('tradeID')
                        
                        print(f"🎯 TAKE PROFIT HIT: {instrument} @ {price}")
                        print(f"   Trade ID: {trade_id}")
                        say_nonblocking(f"Take profit hit on {instrument} at {price}", voice="Alex")
                        
                    elif transaction_type == 'MARKET_ORDER':
                        # Market order created
                        instrument = data.get('instrument')
                        units = data.get('units')
                        side = "BUY" if float(units) > 0 else "SELL"
                        
                        print(f"📋 MARKET ORDER CREATED: {side} {abs(float(units))} {instrument}")
                        
                    elif transaction_type == 'HEARTBEAT':
                        # Heartbeat to keep connection alive
                        if transaction_count % 50 == 0:  # Print occasionally
                            print(f"💓 Transaction stream heartbeat ({transaction_count} transactions)")
                    
                    else:
                        # Other transaction types
                        print(f"📨 Transaction: {transaction_type} - {data}")
                    
                    transaction_count += 1
                    
                    # Call callback function if provided
                    if callback:
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"⚠️  Callback error: {e}")
                    
                    # Yield transaction data for generator usage
                    yield data
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️  Processing error: {e}")
                    continue
        
        print(f"\n✅ Transaction streaming completed. Total events: {transaction_count}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Transaction streaming stopped by user. Total events: {transaction_count}")
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT ERROR: Transaction stream timed out")
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Lost connection to OANDA transaction stream")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    finally:
        if 'response' in locals():
            response.close()


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



def get_oanda_data(credentials, instrument='USD_CAD', granularity='S5', hours=5, rows=5000):
    """
    Connect to OANDA live fxtrade environment and fetch real market data
    
    Parameters:
    -----------
    credentials : dict
        Dictionary containing 'api_key' and 'account_id'
    instrument : str, default='USD_CAD'
        Currency pair to fetch
    granularity : str, default='S5'
        Time granularity (S5 = 5 seconds)
    hours : int, default=10
        Number of hours of historical data to fetch
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with real market data
    """
    
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')
    
    # LIVE OANDA API URL (NOT practice!)
    BASE_URL = "https://api-fxtrade.oanda.com"
    
    print("🔴 CONNECTING TO OANDA LIVE FXTRADE ENVIRONMENT")
    print("=" * 55)
    print("⚠️  WARNING: This will connect to LIVE market data")
    print(f"📊 Requesting: {instrument} | {granularity} | Last {hours} hours")
    print("=" * 55)
    
    # Validate inputs
    if not api_key or api_key == "your_live_api_key_here":
        print("❌ ERROR: Live API key is required!")
        print("\n🔧 TO GET YOUR LIVE OANDA CREDENTIALS:")
        print("1. Log into your OANDA account at: https://www.oanda.com/")
        print("2. Go to 'Manage API Access' in account settings")
        print("3. Generate a Personal Access Token")
        print("4. Copy your Account ID from account overview")
        print("\n💡 USAGE:")
        print("live_data = connect_oanda_live({")
        print("    'api_key': 'your_actual_api_key',")
        print("    'account_id': 'your_actual_account_id'")
        print("})")
        return None
    
    if not account_id or account_id == "your_live_account_id_here":
        print("❌ ERROR: Live Account ID is required!")
        return None
    
    # Headers for API request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # Calculate count based on granularity and hours
    if granularity == 'S5':
        count = min(hours * 60 * 12, rows)  # 12 five-second intervals per minute, max 5000
    elif granularity == 'S10':
        count = min(hours * 60 * 6, rows)   # 6 ten-second intervals per minute
    elif granularity == 'M1':
        count = min(hours * 60, rows)       # 60 one-minute intervals per hour
    elif granularity == 'M5':
        count = min(hours * 12, rows)       # 12 five-minute intervals per hour
    else:
        count = min(7200, rows)  # Default fallback

    # API endpoint for historical candles
    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    
    # Parameters for the request
    params = {
        'count': count,
        'granularity': granularity#,
        # 'price': 'MBA',  # Mid, Bid, Ask prices
        # 'includeFirst': 'true'
    }
    
    try:
        print(f"🌐 Making API request to OANDA live servers...")
        print(f"   URL: {url}")
        print(f"   Params: {params}")
        
        # Make the API request
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Check for HTTP errors
        if response.status_code == 401:
            print("❌ AUTHENTICATION ERROR (401)")
            print("   • Check your API key is correct")
            print("   • Ensure your API key has proper permissions")
            print("   • Verify you're using the live account API key")
            return None
        elif response.status_code == 403:
            print("❌ FORBIDDEN ERROR (403)")
            print("   • Your account may not have API access enabled")
            print("   • Check if your account is verified and funded")
            return None
        elif response.status_code == 404:
            print("❌ NOT FOUND ERROR (404)")
            print(f"   • Check instrument name: {instrument}")
            print(f"   • Check granularity: {granularity}")
            return None
        elif response.status_code != 200:
            print(f"❌ HTTP ERROR {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        # Parse JSON response
        data = response.json()
        
        if 'candles' not in data:
            print("❌ ERROR: No candles data in response")
            print(f"Response: {data}")
            return None
        
        candles = data['candles']
        print(f"✅ Successfully received {len(candles)} candles from OANDA live")
        
        # Convert to DataFrame
        market_data = []
        for candle in candles:
            # Convert timestamp to New York timezone and remove timezone info
            timestamp = pd.to_datetime(candle['time'])
            # Convert to New York timezone
            timestamp = timestamp.tz_convert('America/New_York')
            # Remove timezone info (localize to None)
            timestamp = timestamp.tz_localize(None)
            
            # Extract OHLC data
            mid = candle.get('mid', {})
            bid = candle.get('bid', {})
            ask = candle.get('ask', {})
            
            if not mid:
                continue  # Skip if no mid prices
            
            # Get prices
            open_price = float(mid['o'])
            high_price = float(mid['h'])
            low_price = float(mid['l'])
            close_price = float(mid['c'])
            
            bid_price = float(bid.get('c', close_price - 0.0001))
            ask_price = float(ask.get('c', close_price + 0.0001))
            
            # Calculate spread in pips (for USD/CAD, 1 pip = 0.0001)
            spread_pips = (ask_price - bid_price) * 10000
            
            market_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'mid': close_price,
                'bid': bid_price,
                'ask': ask_price,
                'volume': candle.get('volume', 0),
                'spread_pips': round(spread_pips, 1),
                'complete': candle.get('complete', True)
            })
        
        if not market_data:
            print("❌ ERROR: No valid market data received")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(market_data)
        
        # Add price column for compatibility with EMA functions
        df['price'] = df['close']
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n📊 LIVE MARKET DATA SUMMARY:")
        print(f"   • Instrument: {instrument}")
        print(f"   • Granularity: {granularity}")
        print(f"   • Total candles: {len(df):,}")
        print(f"   • Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   • Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        print(f"   • Current price: {df['close'].iloc[-1]:.5f}")
        print(f"   • Average spread: {df['spread_pips'].mean():.1f} pips")
        
        # # Show latest data
        # print(f"\n📈 LATEST 3 CANDLES:")
        # latest_cols = ['timestamp', 'open', 'high', 'low', 'close', 'bid', 'ask', 'spread_pips']
        # print(df[latest_cols].tail(3).to_string(index=False, float_format='%.5f'))
        

        # return the dataframe with timestamp and  price columns
        return df[['timestamp', 'price']]
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None
        




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

