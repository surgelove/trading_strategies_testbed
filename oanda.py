
import requests

def get_pip_value(credentials,instrument_name):

    url = credentials.get('url')
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')

    endpoint = f"{url}/v3/accounts/{account_id}/instruments"
    headers = {'Authorization': f'Bearer {api_key}'}
    params = None
    data = None

    try:
        response = requests.get(endpoint, headers=headers, params=params, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        instruments = response.json()['instruments']
        
        for instrument in instruments:
            if instrument['name'] == instrument_name:
                # Calculate pip value based on pip location
                pip_location = instrument.get('pipLocation', -4)  # Default to -4 if not found
                pip_value = 10 ** pip_location
                
                print(f"Instrument: {instrument_name}")
                print(f"Pip Location: {pip_location}")
                print(f"Pip Value: {pip_value}")
                
                return pip_value
                
        print(f"Instrument {instrument_name} not found")
        return None # Instrument not found
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def get_instrument_precision(credentials, instrument_name):

    """
    Retrieves the display precision (decimal places) for a financial instrument from OANDA.

    Makes an authenticated request to the OANDA API to get instrument details and 
    returns the number of decimal places used for price display.

    Args:
        credentials (dict): Dictionary containing 'api_key' and 'account_id'
        instrument_name (str): The instrument name (e.g., 'EUR_USD', 'USD_CAD').

    Returns:
        int: Number of decimal places for price display, or None if instrument not found or error occurs.
    """
    
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')
    url = "https://api-fxtrade.oanda.com"

    endpoint = f"{url}/v3/accounts/{account_id}/instruments"
    headers = {'Authorization': f'Bearer {api_key}'}
    params = None
    data = None
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        instruments = response.json()['instruments']
        
        for instrument in instruments:
            if instrument['name'] == instrument_name:
                return instrument['displayPrecision']
                
        return None # Instrument not found
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
def cancel_oanda_orders(credentials, instrument, side=0):
    """
    Cancel pending orders for a specific instrument and side while preserving 
    existing trades' stop loss and trailing stop loss orders.
    
    Parameters:
    -----------
    credentials : dict
        Dictionary containing 'token' and 'account_id'
    instrument : str
        Trading instrument (e.g., 'USD_CAD')
    side : int
        Order side to cancel: 1 for BUY, -1 for SELL, 0 for both (default 0)
    
    Returns:
    --------
    dict: Summary of cancelled orders and preserved orders
    """
    
    token = credentials.get('token')
    account_id = credentials.get('account_id')
    
    if not token or not account_id:
        return {"error": "Missing credentials: token and account_id required"}
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    base_url = "https://api-fxtrade.oanda.com/v3"
    
    try:
        print(f"🔍 Checking pending orders for {instrument} (side: {side})...")
        
        # Get all pending orders
        orders_url = f"{base_url}/accounts/{account_id}/pendingOrders"
        orders_response = requests.get(orders_url, headers=headers)
        orders_response.raise_for_status()
        
        orders_data = orders_response.json()
        all_orders = orders_data.get('orders', [])
        
        if not all_orders:
            print(f"ℹ️  No pending orders found for account")
            return {"cancelled": 0, "preserved": 0, "orders": []}
        
        # Filter orders for the specific instrument
        instrument_orders = [order for order in all_orders if order.get('instrument') == instrument]
        
        if not instrument_orders:
            print(f"ℹ️  No pending orders found for {instrument}")
            return {"cancelled": 0, "preserved": 0, "orders": []}
        
        print(f"📋 Found {len(instrument_orders)} pending order(s) for {instrument}")
        
        # Categorize orders
        orders_to_cancel = []
        orders_to_preserve = []
        
        # Order types that should be preserved (position-related)
        preserve_order_types = {
            'STOP_LOSS',
            'TRAILING_STOP_LOSS',
            'TAKE_PROFIT',
            'STOP_LOSS_ORDER',
            'TRAILING_STOP_LOSS_ORDER',
            'TAKE_PROFIT_ORDER'
        }
        
        for order in instrument_orders:
            order_type = order.get('type', '')
            order_id = order.get('id')
            order_units = float(order.get('units', '0'))
            
            # Check if this is a position-related order (has linkedOrderID or tradeID)
            is_position_related = (
                order.get('linkedOrderID') is not None or 
                order.get('tradeID') is not None or
                order_type in preserve_order_types
            )
            
            if is_position_related:
                orders_to_preserve.append(order)
                side_text = "BUY" if order_units > 0 else "SELL"
                print(f"   🛡️  Preserving {order_type} order (ID: {order_id}) - {side_text} position protection")
            else:
                # Check if order matches the specified side
                order_side_matches = False
                
                if side == 0:  # Both sides
                    order_side_matches = True
                elif side == 1 and order_units > 0:  # BUY side
                    order_side_matches = True
                elif side == -1 and order_units < 0:  # SELL side
                    order_side_matches = True
                
                if order_side_matches:
                    orders_to_cancel.append(order)
                    side_text = "BUY" if order_units > 0 else "SELL"
                    price = order.get('price', 'Market')
                    print(f"   🎯 Will cancel {order_type} order (ID: {order_id}) - {side_text} @ {price}")
                else:
                    orders_to_preserve.append(order)
                    side_text = "BUY" if order_units > 0 else "SELL"
                    print(f"   ⏭️  Skipping {order_type} order (ID: {order_id}) - {side_text} (different side)")
        
        # Cancel the identified orders
        cancelled_orders = []
        cancelled_count = 0
        
        if orders_to_cancel:
            print(f"\n🔄 Cancelling {len(orders_to_cancel)} pending order(s)...")
            
            for order in orders_to_cancel:
                order_id = order.get('id')
                order_type = order.get('type')
                order_units = float(order.get('units', '0'))
                
                cancel_url = f"{base_url}/accounts/{account_id}/orders/{order_id}/cancel"
                cancel_response = requests.put(cancel_url, headers=headers)
                
                if cancel_response.status_code == 200:
                    cancelled_count += 1
                    side_text = "BUY" if order_units > 0 else "SELL"
                    price = order.get('price', 'Market')
                    cancelled_orders.append({
                        'id': order_id,
                        'type': order_type,
                        'side': side_text,
                        'price': price,
                        'units': str(order_units)
                    })
                    print(f"   ✅ Cancelled {order_type} order (ID: {order_id}) - {side_text} @ {price}")
                else:
                    print(f"   ❌ Failed to cancel order {order_id}: {cancel_response.text}")
        else:
            print(f"\n ℹ️  No orders to cancel for {instrument} on specified side")
        
        # Summary
        preserved_count = len(orders_to_preserve)
        
        side_text = "BOTH" if side == 0 else ("BUY" if side == 1 else "SELL")
        print(f"\n📊 Summary for {instrument} ({side_text} side):")
        print(f"   ✅ Cancelled: {cancelled_count} pending orders")
        print(f"   🛡️  Preserved: {preserved_count} position-related/other-side orders")
        print(f"   📈 Existing trades and their stop losses remain untouched")
        
        return {
            "success": True,
            "instrument": instrument,
            "side": side,
            "cancelled": cancelled_count,
            "preserved": preserved_count,
            "cancelled_orders": cancelled_orders,
            "preserved_orders": [{
                'id': order.get('id'),
                'type': order.get('type'),
                'units': order.get('units'),
                'linkedOrderID': order.get('linkedOrderID'),
                'tradeID': order.get('tradeID')
            } for order in orders_to_preserve]
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

def create_oanda_orders(credentials, instrument, side, 
                        order_type="MARKET", distance=None, 
                        stop_loss=None, trailing_stop=None,
                        position_size_percent=0.95, entries=1, entries_distance=5,
                        cancel_existing=True):
    """
    Improved OANDA order creation function using API's available units
    
    Parameters:
    -----------
    credentials : dict
        Dictionary containing 'api_key' and 'account_id'
    instrument : str
        Trading instrument (e.g., 'USD_CAD')
    side : int
        Order direction: 1 for BUY, -1 for SELL, 0 for both
    order_type : str
        "MARKET" for immediate execution, "STOP" for stop orders, or "LIMIT" for limit orders
    distance : float
        Distance in pips from current price (only for STOP and LIMIT orders)
    stop_loss : float


        Stop loss distance in pips (e.g., 20 means 20 pips stop loss)
        - For BUY orders: stop loss placed BELOW entry price
        - For SELL orders: stop loss placed ABOVE entry price
    trailing_stop : float
        Trailing stop distance in pips (e.g., 15 means 15 pips trailing stop)
        - For BUY orders: trailing stop follows BELOW price
        - For SELL orders: trailing stop follows ABOVE price
    position_size_percent : float
        Percentage of available units to use (default 0.95 for 95%)
    entries : int
        Number of entry orders to create (default 1)
    entries_distance : float
        Distance in pips between multiple entries (default 5)
    cancel_existing : bool
        If True, cancel existing pending orders for the specified instrument and side (default False)
    """
    
    api_key = credentials.get('api_key')
    account_id = credentials.get('account_id')
    
    if not api_key or not account_id:
        return {"error": "Missing credentials: api_key and account_id required"}
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    base_url = "https://api-fxtrade.oanda.com/v3"
    
    try:
        # Cancel existing orders first (if requested) - BEFORE calculating available units
        # This ensures that pending orders don't affect the available units calculation
        if cancel_existing:
            print(f"🔄 Canceling existing pending orders for {instrument}...")
            
            # Get all pending orders
            orders_url = f"{base_url}/accounts/{account_id}/pendingOrders"
            orders_response = requests.get(orders_url, headers=headers)
            
            if orders_response.status_code == 200:
                orders_data = orders_response.json()
                orders_to_cancel = []
                
                for order in orders_data.get('orders', []):
                    if order.get('instrument') == instrument:
                        order_units = float(order.get('units', 0))
                        
                        # Check if order matches the specified side
                        if side == 1 and order_units > 0:  # BUY side
                            orders_to_cancel.append(order)
                        elif side == -1 and order_units < 0:  # SELL side
                            orders_to_cancel.append(order)
                        elif side == 0:  # Both sides
                            orders_to_cancel.append(order)
                
                # Cancel matching orders
                cancelled_count = 0
                for order in orders_to_cancel:
                    order_id = order.get('id')
                    cancel_url = f"{base_url}/accounts/{account_id}/orders/{order_id}/cancel"
                    cancel_response = requests.put(cancel_url, headers=headers)
                    
                    if cancel_response.status_code == 200:
                        cancelled_count += 1
                        order_units = float(order.get('units', 0))
                        side_text = "BUY" if order_units > 0 else "SELL"
                        print(f"   ✅ Cancelled {side_text} {order.get('type')} order (ID: {order_id})")
                    else:
                        print(f"   ❌ Failed to cancel order {order_id}: {cancel_response.text}")
                
                if cancelled_count > 0:
                    print(f"✅ Cancelled {cancelled_count} existing orders for {instrument}")
                else:
                    print(f"ℹ️  No existing orders found for {instrument} on the specified side")
            else:
                print(f"⚠️  Could not retrieve pending orders: {orders_response.text}")
        
        sleep(1)

        # Get current prices AND available units using includeUnitsAvailable
        pricing_url = f"{base_url}/accounts/{account_id}/pricing"
        params = {
            'instruments': instrument,
            'includeUnitsAvailable': True
        }
        pricing_response = requests.get(pricing_url, headers=headers, params=params)
        pricing_response.raise_for_status()
        print(pricing_response.text)
        
        price_data = pricing_response.json()
        if not price_data.get('prices'):
            return {"error": f"No pricing data available for {instrument}"}
            
        price_info = price_data['prices'][0]
        bid_price = float(price_info['bids'][0]['price'])
        ask_price = float(price_info['asks'][0]['price'])
        
        print(f"Current {instrument} - Bid: {bid_price}, Ask: {ask_price}")
        
        # Get available units directly from OANDA API
        units_available = price_info.get('unitsAvailable', {})
        
        # Extract available units for different order types
        default_units = units_available.get('default', {})
        reduce_only_units = units_available.get('reduceOnly', {})
        
        # Use openOnly units since we're using positionFill: "OPEN_ONLY"
        available_long_temp = float(default_units.get('long', 0))
        available_short_temp = float(default_units.get('short', 0))
        position_long = float(reduce_only_units.get('short', 0))
        position_short = float(reduce_only_units.get('long', 0))

        if position_long:
            available_long = available_long_temp
            available_short = available_short_temp - position_long
        elif position_short:
            available_long = available_long_temp - position_short
            available_short = available_short_temp
        else:
            available_long = available_long_temp
            available_short = available_short_temp

        print(f"Available units from API - Long: {available_long}, Short: {available_short}")
        
        # Get existing positions for reference
        positions_url = f"{base_url}/accounts/{account_id}/positions/{instrument}"
        positions_response = requests.get(positions_url, headers=headers)
        
        long_units = 0
        short_units = 0
        
        if positions_response.status_code == 200:
            position_data = positions_response.json()
            position = position_data.get('position', {})
            long_units = float(position.get('long', {}).get('units', 0))
            short_units = abs(float(position.get('short', {}).get('units', 0)))
            
            print(f"Existing {instrument} positions - Long: {long_units}, Short: {short_units}")
        
        # Calculate available units for new orders (using configurable percentage for safety margin)
        percent_display = int(position_size_percent * 100)
        
        if side == 1:  # BUY only
            total_buy_units = int(available_long * position_size_percent)
            buy_units_per_entry = total_buy_units // entries
            print(f"Available units for BUY: {total_buy_units} ({percent_display}% of {int(available_long)})")
            print(f"Units per entry: {buy_units_per_entry} (divided into {entries} entries)")
            if buy_units_per_entry <= 0:
                print("⚠️  No available units for BUY trading. Check your margin or existing positions.")
                return {"error": "No available units for BUY trading"}
        elif side == -1:  # SELL only
            total_sell_units = int(available_short * position_size_percent)
            sell_units_per_entry = total_sell_units // entries
            print(f"Available units for SELL: {total_sell_units} ({percent_display}% of {int(available_short)})")
            print(f"Units per entry: {sell_units_per_entry} (divided into {entries} entries)")
            if sell_units_per_entry <= 0:
                print("⚠️  No available units for SELL trading. Check your margin or existing positions.")
                return {"error": "No available units for SELL trading"}
        else:  # Both BUY and SELL (side == 0)
            total_buy_units = int(available_long * position_size_percent)
            total_sell_units = int(available_short * position_size_percent)
            buy_units_per_entry = total_buy_units // entries
            sell_units_per_entry = total_sell_units // entries
            print(f"Available units - BUY: {total_buy_units} ({percent_display}% of {int(available_long)}), SELL: {total_sell_units} ({percent_display}% of {int(available_short)})")
           
            print(f"Units per entry - BUY: {buy_units_per_entry}, SELL: {sell_units_per_entry} (divided into {entries} entries)")
            if buy_units_per_entry <= 0 and sell_units_per_entry <= 0:
                print("⚠️  No available units for trading. Check your margin or existing positions.")
                return {"error": "No available units for trading"}
        
        # Calculate pip value
        pip_value = 0.01 if 'JPY' in instrument else 0.0001
        
        orders_created = []
        
        # Create BUY orders
        if side == 1 or side == 0:
            # Use available units for BUY orders
            if side == 1:  # BUY only
                use_buy_units = buy_units_per_entry
            else:  # Both BUY and SELL (side == 0)
                use_buy_units = buy_units_per_entry
            
            if use_buy_units > 0:
                # Create multiple BUY entries
                for entry_num in range(entries):
                    buy_order = {
                        "order": {
                            "units": str(use_buy_units),
                            "instrument": instrument,
                            "timeInForce": "GTC" if order_type in ["LIMIT", "STOP"] else "FOK",
                            "type": order_type,
                            "positionFill": "OPEN_ONLY"
                        }
                    }
                
                    # Add price for LIMIT and STOP orders
                    if order_type in ["LIMIT", "STOP"] and distance:
                        # Calculate price for this entry (first entry uses base distance, others are spaced by entries_distance)
                        entry_distance = distance + (entry_num * entries_distance)
                        
                        if order_type == "STOP":
                            # For STOP BUY orders, price should be ABOVE current ask (breakout)
                            buy_price = round(ask_price + (entry_distance * pip_value), 5)
                        else:  # LIMIT
                            # For LIMIT BUY orders, price should be BELOW current ask (better price)
                            buy_price = round(ask_price - (entry_distance * pip_value), 5)
                        
                        buy_order["order"]["price"] = str(buy_price)
                        print(f"BUY {order_type.lower()} order #{entry_num + 1} at: {buy_price} (Units: {use_buy_units})")
                    else:
                        print(f"BUY {order_type} order #{entry_num + 1} (Units: {use_buy_units})")
                    
                    # Add stop loss and trailing stop (convert pips to distance)
                    # For BUY orders, stop loss should be BELOW the entry price
                    if stop_loss:
                        # For SELL orders, we need to calculate the actual stop loss price
                        # The stop loss should be ABOVE the sell price
                        if order_type == "MARKET":
                            # For market orders, use current bid price + stop loss distance
                            stop_loss_price = bid_price - (stop_loss * pip_value)
                        else:
                            # For limit/stop orders, use the order price + stop loss distance
                            stop_loss_price = buy_price - (stop_loss * pip_value)
                        
                        buy_order["order"]["stopLossOnFill"] = {"price": str(round(stop_loss_price, 5))}
                        print(f"   Stop Loss: {stop_loss} pips ABOVE entry at {stop_loss_price}")
                    if trailing_stop:
                        # Convert pips to distance format (distance is always positive)
                        trailing_stop_distance = trailing_stop * pip_value
                        buy_order["order"]["trailingStopLossOnFill"] = {"distance": str(trailing_stop_distance)}
                        print(f"   Trailing Stop: {trailing_stop} pips BELOW entry ({trailing_stop_distance})")
                    
                    print('---------------------')
                    print(buy_order)
                    print('---------------------')# Send order

                    order_url = f"{base_url}/accounts/{account_id}/orders"
                    response = requests.post(order_url, headers=headers, json=buy_order)
                    
                    if response.status_code == 201:
                        result = response.json()
                        orders_created.append({"BUY": result})
                        print(f"✅ BUY order #{entry_num + 1} created successfully!")
                        
                        # Check if market order was filled
                        if 'orderFillTransaction' in result:
                            fill_info = result['orderFillTransaction']
                            print(f"   Filled at: {fill_info.get('price')}")
                            print(f"   Units: {fill_info.get('units')}")
                    else:
                        error_msg = f"BUY order #{entry_num + 1} failed: {response.text}"
                        print(f"❌ {error_msg}")
                        return {"error": error_msg}
            else:
                print("⚠️  No available units for BUY orders")
        
        # Create SELL orders
        if side == -1 or side == 0:
            # Use available units for SELL orders
            if side == -1:  # SELL only
                use_sell_units = sell_units_per_entry
            else:  # Both BUY and SELL (side == 0)
                use_sell_units = sell_units_per_entry
            
            if use_sell_units > 0:
                # Create multiple SELL entries
                for entry_num in range(entries):
                    sell_order = {
                        "order": {
                            "units": str(-use_sell_units),  # Negative for SELL
                            "instrument": instrument,
                            "timeInForce": "GTC" if order_type in ["LIMIT", "STOP"] else "FOK",
                            "type": order_type,
                                                       "positionFill": "OPEN_ONLY"
                        }
                    }
                
                    # Add price for LIMIT and STOP orders
                    if order_type in ["LIMIT", "STOP"] and distance:
                        # Calculate price for this entry (first entry uses base distance, others are spaced by entries_distance)
                        entry_distance = distance + (entry_num * entries_distance)
                        
                        if order_type == "STOP":
                            # For STOP SELL orders, price should be BELOW current bid (breakdown)
                            sell_price = round(bid_price - (entry_distance * pip_value), 5)
                        else:  # LIMIT
                            # For LIMIT SELL orders, price should be ABOVE current bid (better price)
                            sell_price = round(bid_price + (entry_distance * pip_value), 5)
                        
                        sell_order["order"]["price"] = str(sell_price)
                        print(f"SELL {order_type.lower()} order #{entry_num + 1} at: {sell_price} (Units: {use_sell_units})")
                    else:
                        print(f"SELL {order_type} order #{entry_num + 1} (Units: {use_sell_units})")
                    
                    # Add stop loss and trailing stop (convert pips to distance)
                    # For SELL orders, stop loss should be ABOVE the entry price
                    if stop_loss:
                        # For SELL orders, we need to calculate the actual stop loss price
                        # The stop loss should be ABOVE the sell price
                        if order_type == "MARKET":
                            # For market orders, use current bid price + stop loss distance
                            stop_loss_price = bid_price + (stop_loss * pip_value)
                        else:
                            # For limit/stop orders, use the order price + stop loss distance
                            stop_loss_price = sell_price + (stop_loss * pip_value)
                        
                        sell_order["order"]["stopLossOnFill"] = {"price": str(round(stop_loss_price, 5))}
                        print(f"   Stop Loss: {stop_loss} pips ABOVE entry at {stop_loss_price}")

                    if trailing_stop:
                        # Convert pips to distance format (distance is always positive)
                        trailing_stop_distance = trailing_stop * pip_value
                        sell_order["order"]["trailingStopLossOnFill"] = {"distance": str(trailing_stop_distance)}
                        print(f"   Trailing Stop: {trailing_stop} pips ABOVE entry ({trailing_stop_distance})")
                    
                    print('---------------------')
                    print(sell_order)
                    print('---------------------')

                    # Send order
                    order_url = f"{base_url}/accounts/{account_id}/orders"
                    response = requests.post(order_url, headers=headers, json=sell_order)
                    
                    if response.status_code == 201:
                        result = response.json()
                        orders_created.append({"SELL": result})
                        print(f"✅ SELL order #{entry_num + 1} created successfully!")
                        
                        # Check if market order was filled
                        if 'orderFillTransaction' in result:
                            fill_info = result['orderFillTransaction']
                            print(f"   Filled at: {fill_info.get('price')}")
                            print(f"   Units: {fill_info.get('units')}")
                    else:
                        error_msg = f"SELL order #{entry_num + 1} failed: {response.text}"
                        print(f"❌ {error_msg}")
                        return {"error": error_msg}
            else:
                print("⚠️  No available units for SELL orders")
        
        return orders_created
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    except KeyError as e:
        error_msg = f"Missing required data in API response: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

