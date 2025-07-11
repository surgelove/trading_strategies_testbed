import subprocess
import threading
import time
import json
from collections import deque
import pandas as pd
from datetime import datetime, timedelta
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_notebook, show
import requests
import tkinter as tk
from tkinter import messagebox
import pyttsx3
from live_graph_server import run_server, graph_updater


class Algo:
    """
    Time-based streaming moving average calculator that uses actual timestamps
    instead of fixed number of data points. Supports SMA, EMA, DEMA, and TEMA.
    
    Key Features:
    - Uses time windows (e.g., "5 minutes", "1 hour") instead of row counts
    - Automatically handles irregular time intervals
    - Maintains time-weighted calculations
    - Supports all four MA types with time-based logic
    """

    def __init__(self, base_interval, slow_interval, aspr_interval, peak_interval=None):

        self.base_ema_calc = TimeBasedStreamingMA(base_interval, ma_type='EMA')
        self.base_tema_calc = TimeBasedStreamingMA(base_interval, ma_type='TEMA')
        self.slow_ema_calc = TimeBasedStreamingMA(slow_interval, ma_type='EMA')
        self.slow_tema_calc = TimeBasedStreamingMA(slow_interval, ma_type='TEMA')
        self.aspr_ema_calc = TimeBasedStreamingMA(aspr_interval, ma_type='EMA')
        self.aspr_tema_calc = TimeBasedStreamingMA(aspr_interval, ma_type='TEMA')
        self.peak_ema_calc = TimeBasedStreamingMA(peak_interval, ma_type='EMA')
        self.peak_tema_calc = TimeBasedStreamingMA(peak_interval, ma_type='TEMA')
        self.peak_dema_calc = TimeBasedStreamingMA(peak_interval, ma_type='DEMA')

        # initialize the json that will hold timestamp price and ema values
        self.base_ema_values = []
        self.base_tema_values = []
        self.slow_ema_values = []
        self.slow_tema_values = []
        self.aspr_ema_values = []
        self.aspr_tema_values = []
        self.peak_ema_values = []
        self.peak_tema_values = []
        self.peak_dema_values = []
        self.base_mamplitudes = []
        self.base_pamplitudes = []  # List to hold base_pamplitude values

        self.base_cross_directions = []  # List to hold cross direction
        self.base_cross_prices = []  # List to hold cross price
        self.base_cross_prices_up = []  # List to hold cross up prices
        self.base_cross_prices_down = []  # List to hold cross down prices

        self.base_min_prices = []  # List to hold minimum prices since cross down
        self.base_max_prices = []  # List to hold maximum prices since cross up
        self.base_min_price = None  # keeps track of the minimum price
        self.base_max_price = None  # keeps track of the maximum price
        self.base_min_price_latest = None  # keeps track of the latest minimum price
        self.base_max_price_latest = None  # keeps track of the latest maximum price

        self.aspr_min_prices = []  # List to hold minimum prices since cross down for aspr_ema
        self.aspr_max_prices = []
        self.aspr_min_price = None
        self.aspr_max_price = None

        self.base_travels = []  # List to hold travel values
        self.base_travel = None  # keeps track of the travel from min to max after cross up

        self.base_mamplitude_enough = False  # Flag to indicate if amplitude is greater than 0.02

        self.base_cross_price_previous = None  # Previous cross price
        self.base_pamplitude_enough = False  # Flag to indicate if amplitude is greater than 0.002

        self.base_cross_direction_previous = 0  # Previous cross direction
        
        # Add tracking for maximum amplitudes
        self.base_mamplitude_max = 0  # Track maximum base_mamplitude since last reset
        self.base_pamplitude_max = 0  # Track maximum base_pamplitude since last reset

        self.base_mamplitude_threshold = 0.002  # Threshold for base_mamplitude to be considered significant
        self.base_pamplitude_threshold = 0.001  # Threshold for base_pamplitude to be considered significant
        self.peak_travel_threshold = 0.04  # Threshold for peak travel to be considered significant

        self.peak_cross_price_previous = None  # Previous price for peak cross detection
        # self.peak_price_above = False  # Price above peak_ema for peak cross detection
        # self.peak_price_below = False  # Price below peak_ema for peak cross detection
        self.peak_price_crossed_up = False  # Flag to indicate if price crossed peak_ema
        self.peak_price_crossed_dn = False  # Flag to indicate if price crossed peak_ema

        self.peak_cross_price_dn = None  # Price when peak cross down occurs
        self.peak_cross_price_up = None  # Price when peak cross up occurs

        self.peak_travel = 0

        self.peak_cross_direction = None

    def process_row(self, timestamp, price, precision, say):

        threshold = 0.00005  # Threshold for ignoring min/max prices close to current price
        if not self.base_cross_price_previous:
            self.base_cross_price_previous = price

        # Calculate EMA and TEMA for the current price
        base_ema = round(self.base_ema_calc.add_data_point(timestamp, price), precision)
        base_tema = round(self.base_tema_calc.add_data_point(timestamp, price), precision)
        self.base_ema_values.append(base_ema)  # EMA value
        self.base_tema_values.append(base_tema)  # TEMA value

        aspr_ema = round(self.aspr_ema_calc.add_data_point(timestamp, price), precision)
        aspr_tema = round(self.aspr_tema_calc.add_data_point(timestamp, price), precision)
        self.aspr_ema_values.append(aspr_ema)
        self.aspr_tema_values.append(aspr_tema)

        peak_ema = round(self.peak_ema_calc.add_data_point(timestamp, price), precision)
        peak_tema = round(self.peak_tema_calc.add_data_point(timestamp, price), precision)
        peak_dema = round(self.peak_dema_calc.add_data_point(timestamp, price), precision)
        self.peak_ema_values.append(peak_ema)
        self.peak_tema_values.append(peak_tema)
        self.peak_dema_values.append(peak_dema)

        # when peak_tema crosses peak_ema, start detectng travel from one direction to the other in percentage
        if self.peak_cross_price_previous:
            new_travel = abs((self.peak_cross_price_previous - price) / self.peak_cross_price_previous * 100)
            if new_travel > self.peak_travel:
                self.peak_travel = new_travel
        else:
            self.peak_travel = 0
        if len(self.peak_ema_values) > 1 and len(self.peak_tema_values) > 1:
            if (self.peak_tema_values[-1] > self.peak_ema_values[-1] and self.peak_tema_values[-2] <= self.peak_ema_values[-2]):
                self.peak_cross_price_previous = price
                self.peak_travel = 0
                # self.peak_price_below = False
                self.peak_cross_direction = 1   
            elif (self.peak_tema_values[-1] < self.peak_ema_values[-1] and self.peak_tema_values[-2] >= self.peak_ema_values[-2]):
                self.peak_cross_price_previous = price
                self.peak_travel = 0
                # self.peak_price_above = False
                self.peak_cross_direction = -1

        # when price crosses peak_ema, detect the direction
        peak_cross_price_up = None
        peak_cross_price_dn = None
        if len(self.peak_dema_values) > 1:
            # Fix: Compare price positions relative to peak_ema, not peak_ema relative to price
            if price > self.peak_dema_values[-1]:
                self.peak_price_crossed_dn = False                 # Since we are above the peak_ema, reset the fact that we crossed down, so we can go below it again
                if self.peak_cross_direction == -1:                         # if we are going down
                    if self.peak_travel > self.peak_travel_threshold:       # if the price traveled enough from the last cross
                        if not self.peak_price_crossed_up:         # if we haven't crossed up the peak_dema already
                            if not self.peak_cross_price_up or price < self.peak_cross_price_up:
                                peak_cross_price_up = price                     # signal up at that price so a bar above should be drawn 
                                
                                self.peak_cross_price_up = peak_cross_price_up      # remember the price when we crossed up
                                self.peak_price_crossed_up = True      # remember that we crossed up the peak_ema
                    else:
                        ...
                        # if self.peak_cross_price_up and price < self.peak_cross_price_up:
                        #     peak_cross_price_up = price
                        #     self.peak_cross_price_up = peak_cross_price_up


            elif price < self.peak_dema_values[-1]:
                self.peak_price_crossed_up = False                     # Since we are below the peak_ema, reset the fact that we crossed up, so we can go above it again
                if self.peak_cross_direction == 1:                              # if we are going up   
                    if self.peak_travel > self.peak_travel_threshold:           # if the price traveled enough from the last cross
                        if not self.peak_price_crossed_dn:             # if we haven't crossed down the peak_dema already
                            if not self.peak_cross_price_dn or price > self.peak_cross_price_dn: 
                                peak_cross_price_dn = price                     # signal down at that price so a bar above should be drawn  
                                
                                self.peak_cross_price_dn = peak_cross_price_dn      # remember the price when we crossed down
                                self.peak_price_crossed_dn = True      # remember that we crossed down the peak_ema
                    else:
                        ...
                        # if self.peak_cross_price_dn and price > self.peak_cross_price_dn:
                        #     peak_cross_price_dn = price
                        #     self.peak_cross_price_dn = peak_cross_price_dn



        # when aspr_tema crosses aspr_ema, detect the direction
        aspr_cross_direction = None
        if len(self.aspr_ema_values) > 1 and len(self.aspr_tema_values) > 1:
            if (self.aspr_tema_values[-1] > self.aspr_ema_values[-1] and self.aspr_tema_values[-2] <= self.aspr_ema_values[-2]):
                aspr_cross_direction = 1
            elif (self.aspr_tema_values[-1] < self.aspr_ema_values[-1] and self.aspr_tema_values[-2] >= self.aspr_ema_values[-2]):
                aspr_cross_direction = -1
        
        # every row, calculate the min price of all prices since last cross down
        if self.aspr_min_price is None:
            self.aspr_min_price = price
        if price < self.aspr_min_price:
            self.aspr_min_price = price
        if aspr_cross_direction == 1:  # If last cross was down
            # if max price is too close in % from price itself, ignore it
            if self.aspr_min_price is not None:
                # print number as format 0.0000000 no matter how small
                print(f"{self.aspr_min_price:.8f} {price:.8f}")
                print(f"{round(abs(self.aspr_min_price - price) / price, 8):.8f}")
            if self.aspr_min_price is not None and abs(self.aspr_min_price - price) / price < threshold:
                self.aspr_min_price = None

            else:
                self.aspr_min_prices.append(self.aspr_min_price)  # Append the minimum price since last cross down
                self.aspr_min_price = None  # Reset min price after cross down
        else:
            self.aspr_min_prices.append(None)

        # every row, calculate the max price of all prices since last cross up
        if self.aspr_max_price is None:
            self.aspr_max_price = price
        if price > self.aspr_max_price:
            self.aspr_max_price = price
        if aspr_cross_direction == -1:  # If last cross was up
            # if max price is too close in % from price itself, ignore it
            if self.aspr_max_price is not None:
                print(f"{self.aspr_max_price:.8f} {price:.8f}")
                print(f"{round(abs(self.aspr_max_price - price) / price, 8):.8f}")
            if self.aspr_max_price is not None and abs(self.aspr_max_price - price) / price < threshold:
                self.aspr_max_price = None
            else:
                self.aspr_max_prices.append(self.aspr_max_price)  # Append the maximum price since last cross up
                self.aspr_max_price = None  # Reset max price after cross up
        else:
            self.aspr_max_prices.append(None)

        # ---------------------------------------------
        
        # Calculate the amplitude between EMA and TEMA
        base_mamplitude = None
        base_mamplitude_temp = round(abs(base_ema - base_tema), precision)  # Calculate the amplitude between EMA and TEMA
        # percent of price
        base_mamplitude_current = round((base_mamplitude_temp / price) * 100, precision) if price != 0 else 0
        
        # Only allow base_mamplitude to increase until reset, with maximum cap of 0.1
        if base_mamplitude_current > self.base_mamplitude_max:
            self.base_mamplitude_max = min(base_mamplitude_current, 0.1)  # Cap at 0.1 (10%)
        base_mamplitude = self.base_mamplitude_max
        
        self.base_mamplitudes.append(base_mamplitude)  # Append the amplitude to the list
        if base_mamplitude > self.base_mamplitude_threshold:
            self.base_mamplitude_enough = True  # Set flag if amplitude is greater than 0.002


        base_pamplitude = None
        base_pamplitude_temp = round(abs(price - self.base_cross_price_previous), precision)  # Calculate the amplitude between current price and previous cross price
        # percent of price
        base_pamplitude_current = round((base_pamplitude_temp / price) * 100, precision) if price != 0 else 0
        
        # Only allow base_pamplitude to increase until reset, with maximum cap of 0.1
        if base_pamplitude_current > self.base_pamplitude_max:
            self.base_pamplitude_max = min(base_pamplitude_current, 0.1)  # Cap at 0.1 (10%)
        base_pamplitude = self.base_pamplitude_max
        
        self.base_pamplitudes.append(base_pamplitude)  # Append the amplitude to the list
        if base_pamplitude > self.base_pamplitude_threshold:
            self.base_pamplitude_enough = True

        #  When tema crosses ema, detect the direction
        base_take = False
        base_cross_direction = None
        base_cross_price = None
        base_cross_price_up = None
        base_cross_price_down = None
        if len(self.base_ema_values) > 1 and len(self.base_tema_values) > 1:
            if (self.base_tema_values[-1] > self.base_ema_values[-1] and self.base_tema_values[-2] <= self.base_ema_values[-2]):
                self.base_cross_price_previous = price
                if self.base_cross_direction_previous in [0,-1]:  # If last cross was down
                    if price > base_ema or price > base_tema:  # Check if price is above EMA or TEMA
                        if self.base_mamplitude_enough or self.base_pamplitude_enough:
                            base_cross_price = price
                            base_cross_price_up = price  # Store the price at which the cross occurred
                            if say: say_nonblocking("Cross up detected", voice="Alex")
                            base_take = 1
                        # print('up', cross_price, timestamp, tema_values[-1], ema_values[-1])  # Print timestamp and values
                        else:
                            if say: say_nonblocking("Cross up detected but not enough amplitude", voice="Alex")
                        base_cross_direction = 1
                        print(f'{timestamp} - Price: {price}, EMA: {base_ema}, TEMA: {base_tema}, E base_mamplitude: {self.base_mamplitude_enough}, Cross Direction: {base_cross_direction}')
                        self.base_mamplitude_enough = False  # Reset flag after cross up
                        self.base_pamplitude_enough = False
                        self.base_mamplitude_max = 0  # Reset max base_mamplitude
                        self.base_pamplitude_max = 0  # Reset max base_pamplitude
                        self.base_cross_direction_previous = 1

            elif (self.base_tema_values[-1] < self.base_ema_values[-1] and self.base_tema_values[-2] >= self.base_ema_values[-2]):
                self.base_cross_price_previous = price
                if self.base_cross_direction_previous in [0,1]:  # If last cross was up
                    if price < base_ema or price < base_tema:  # Check if price is below EMA
                        if self.base_mamplitude_enough or self.base_pamplitude_enough:
                            base_cross_price = price
                            base_cross_price_down = price  # Store the price at which the cross occurred
                            if say: say_nonblocking("Cross down detected", voice="Samantha")
                            base_take = -1
                        else:
                            if say: say_nonblocking("Cross down detected but not enough amplitude", voice="Samantha")
                        base_cross_direction = -1
                        print(f'{timestamp } - Price: {price}, EMA: {base_ema}, TEMA: {base_tema}, E base_mamplitude: {self.base_mamplitude_enough}, Cross Direction: {base_cross_direction}')
                        self.base_mamplitude_enough = False
                        self.base_pamplitude_enough = False
                        self.base_mamplitude_max = 0  # Reset max base_mamplitude
                        self.base_pamplitude_max = 0  # Reset max base_pamplitude
                        self.base_cross_direction_previous = -1


        self.base_cross_directions.append(base_cross_direction)  # Append cross direction to the list
        self.base_cross_prices.append(base_cross_price)  # Append cross price to the list
        self.base_cross_prices_up.append(base_cross_price_up)  # Append cross price up to the list
        self.base_cross_prices_down.append(base_cross_price_down)  # Append cross price down to the list

        # when it crosses up, calculate the travel from min to max
        base_travel = None
        if base_cross_direction == 1:  # If last cross was up
            # Calculate the travel from min to max
            if self.base_min_price_latest is not None and self.base_max_price_latest is not None:
                base_travel = round(abs(self.base_max_price_latest - self.base_min_price_latest), precision)  # Travel from min to max
            # convert travel to percentage of min price
            if self.base_min_price_latest is not None and base_travel is not None:
                base_travel = round((base_travel / self.base_min_price_latest) * 100, precision)
        self.base_travels.append(base_travel)  # Append travel value to the list

        # every row, calculate the min price of all prices since last cross down
        if self.base_min_price is None:
            self.base_min_price = price
        if price < self.base_min_price:
            self.base_min_price = price
        if base_cross_direction == 1:  # If last cross was down
            self.base_min_prices.append(self.base_min_price)  # Append the minimum price since last cross down
            self.base_min_price_latest = self.base_min_price  # Update latest min price
            self.base_min_price = None  # Reset min price after cross down
        else:
            self.base_min_prices.append(None)

        # every row, calculate the max price of all prices since last cross up
        if self.base_max_price is None:
            self.base_max_price = price
        if price > self.base_max_price:
            self.base_max_price = price
        if base_cross_direction == -1:  # If last cross was up
            self.base_max_prices.append(self.base_max_price)  # Append the maximum price since last cross up
            self.base_max_price_latest = self.base_max_price  # Update latest max price
            self.base_max_price = None  # Reset max price after cross up
        else:
            self.base_max_prices.append(None)

        return_dict = {
            'timestamp': timestamp,
            'price': price,
            'base_ema': base_ema,
            'base_tema': base_tema,
            'base_mamplitude': base_mamplitude,
            'base_pamplitude': base_pamplitude,
            'base_cross_direction': base_cross_direction,
            'base_cross_price': base_cross_price,
            'base_cross_price_up': base_cross_price_up,
            'base_cross_price_down': base_cross_price_down,
            'peak_cross_direction': self.peak_cross_direction,
            'peak_cross_price_up': peak_cross_price_up,
            'peak_cross_price_dn': peak_cross_price_dn,
            'peak_travel': self.peak_travel,
            'base_min_price': self.base_min_prices[-1],
            'base_max_price': self.base_max_prices[-1],
            'aspr_min_price': self.aspr_min_prices[-1],
            'aspr_max_price': self.aspr_max_prices[-1],
            'Travel': base_travel,
            # 'Take': take
        }

        return take, return_dict
        

def say_nonblocking(text, voice=None):
    """
    Say text on macOS using the 'say' command in a non-blocking way
    
    Parameters:
    -----------
    text : str
        Text to speak
    voice : str, optional
        Voice to use (e.g., 'Alex', 'Samantha', 'Victoria')
    """
    print("Speaking:", text)
    def speak():
        try:
            cmd = ['say']
            if voice:
                cmd.extend(['-v', voice])
            cmd.append(text)
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error with text-to-speech: {e}")
    
    # Run in a separate thread to avoid blocking
    thread = threading.Thread(target=speak, daemon=True)
    thread.start()


class TimeBasedStreamingMA:
    """
    Time-based streaming moving average calculator that uses actual timestamps
    instead of fixed number of data points. Supports SMA, EMA, DEMA, and TEMA.
    
    Key Features:
    - Uses time windows (e.g., "5 minutes", "1 hour") instead of row counts
    - Automatically handles irregular time intervals
    - Maintains time-weighted calculations
    - Supports all four MA types with time-based logic
    """
    
    def __init__(self, time_window, ma_type='SMA', alpha=None):
        """
        Initialize the time-based streaming moving average calculator.
        
        Parameters:
        -----------
        time_window : str or timedelta
            Time window for calculations (e.g., '5min', '1H', '30s')
            Can be pandas timedelta string or datetime.timedelta object
        ma_type : str
            Type of moving average: 'SMA', 'EMA', 'DEMA', 'TEMA'
        alpha : float, optional
            Smoothing factor for EMA. If None, calculated based on time window
        """
        self.ma_type = ma_type.upper()
        
        # Validate MA type
        if self.ma_type not in ['SMA', 'EMA', 'DEMA', 'TEMA']:
            raise ValueError("ma_type must be one of: 'SMA', 'EMA', 'DEMA', 'TEMA'")
        
        # Convert time window to timedelta
        if isinstance(time_window, str):
            # Handle common abbreviations and deprecated formats
            time_window_str = time_window.replace('H', 'h')  # Fix deprecated 'H' format
            self.time_window = pd.Timedelta(time_window_str)
        elif isinstance(time_window, timedelta):
            self.time_window = pd.Timedelta(time_window)
        else:
            raise ValueError("time_window must be a string (e.g., '5min') or timedelta object")
        
        # Store original time window specification
        self.time_window_str = str(time_window)
        
        # Calculate alpha for EMA-based calculations
        if alpha is None:
            # Convert time window to approximate number of periods for alpha calculation
            # Assume 1-minute base period for alpha calculation
            minutes = self.time_window.total_seconds() / 60
            equivalent_periods = max(1, minutes)  # At least 1 period
            self.alpha = 2.0 / (equivalent_periods + 1)
        else:
            if not (0 < alpha < 1):
                raise ValueError("alpha must be between 0 and 1")
            self.alpha = alpha
        
        # Initialize data storage - we need to keep all data within time window for SMA
        self.data_points = deque()  # Store (timestamp, price) tuples
        self.timestamps = deque()   # Store just timestamps for quick access
        
        # For EMA, DEMA, TEMA - maintain running calculations
        if self.ma_type != 'SMA':
            self.ema1 = None  # First EMA
            self.ema2 = None  # Second EMA (for DEMA, TEMA)
            self.ema3 = None  # Third EMA (for TEMA)
            self.initialized = False
            self.last_timestamp = None
        
        self.data_count = 0
        
    def _clean_old_data(self, current_timestamp):
        """Remove data points older than the time window."""
        cutoff_time = current_timestamp - self.time_window
        
        # Remove old data points
        while self.data_points and self.data_points[0][0] < cutoff_time:
            self.data_points.popleft()
            if self.timestamps:
                self.timestamps.popleft()
    
    def _calculate_time_weight(self, current_timestamp, last_timestamp):
        """Calculate time-based weight for EMA calculations."""
        if last_timestamp is None:
            return self.alpha
        
        # Calculate time elapsed in seconds
        time_elapsed = (current_timestamp - last_timestamp).total_seconds()
        
        # Handle edge cases
        if time_elapsed <= 0:
            return self.alpha  # No time elapsed, use base alpha
        
        # Assume base interval for alpha calculation (e.g., 60 seconds)
        base_interval = 60.0  # 1 minute
        
        # Adjust alpha based on actual time elapsed
        time_factor = time_elapsed / base_interval
        
        # Prevent issues with very small alpha values and large time factors
        if self.alpha <= 0 or self.alpha >= 1:
            return self.alpha
        
        # Apply time-weighted alpha: more time elapsed = more weight to new data
        try:
            adjusted_alpha = 1 - (1 - self.alpha) ** time_factor
            return min(1.0, max(0.0, adjusted_alpha))  # Clamp between 0 and 1
        except (ZeroDivisionError, OverflowError, ValueError):
            # Fallback to base alpha if calculation fails
            return self.alpha
    
    def add_data_point(self, timestamp, price):
        """
        Add a new data point and calculate the updated time-based moving average.
        
        Parameters:
        -----------
        timestamp : datetime or str
            Timestamp of the data point
        price : float
            Price value
            
        Returns:
        --------
        dict
            Dictionary containing current MA, time window info, and metadata
        """
        # Convert timestamp to datetime if it's a string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        self.data_count += 1
        
        # Clean old data points outside time window
        self._clean_old_data(timestamp)
        
        # Add new data point
        self.data_points.append((timestamp, price))
        self.timestamps.append(timestamp)
        
        if self.ma_type == 'SMA':
            return self._calculate_time_sma(timestamp, price)
        elif self.ma_type == 'EMA':
            return self._calculate_time_ema(timestamp, price)
        elif self.ma_type == 'DEMA':
            return self._calculate_time_dema(timestamp, price)
        elif self.ma_type == 'TEMA':
            return self._calculate_time_tema(timestamp, price)
    
    def _calculate_time_sma(self, timestamp, price, return_full_window=False):
        """Calculate time-based Simple Moving Average."""
        # Calculate SMA using all data points within time window
        if len(self.data_points) == 0:
            current_ma = price
        else:
            total_price = sum(p for t, p in self.data_points)
            current_ma = total_price / len(self.data_points)
        
        if return_full_window:
            return {
                'timestamp': timestamp,
                'price': price,
                'moving_average': current_ma,
                'data_points_count': len(self.data_points),
                'time_window': self.time_window_str,
                'window_start': self.timestamps[0] if self.timestamps else timestamp,
                'window_end': timestamp,
                'time_span_actual': (timestamp - self.timestamps[0]).total_seconds() if self.timestamps else 0,
                'time_span_target': self.time_window.total_seconds(),
                'is_full_window': (timestamp - self.timestamps[0]) >= self.time_window if self.timestamps else False,
                'ma_type': 'Time-SMA'
            }
        return current_ma
    
    def _calculate_time_ema(self, timestamp, price, return_full_window=False):
        """Calculate time-based Exponential Moving Average."""
        if not self.initialized:
            # Initialize with first price
            self.ema1 = price
            self.initialized = True
            self.last_timestamp = timestamp
            time_weight = self.alpha
        else:
            # Calculate time-adjusted alpha
            time_weight = self._calculate_time_weight(timestamp, self.last_timestamp)
            # EMA calculation with time weighting
            self.ema1 = time_weight * price + (1 - time_weight) * self.ema1
            self.last_timestamp = timestamp
        
        if return_full_window:
            return {
                'timestamp': timestamp,
                'price': price,
                'moving_average': self.ema1,
                'data_points_count': self.data_count,
                'time_window': self.time_window_str,
                'window_start': self.timestamps[0] if self.timestamps else timestamp,
                'window_end': timestamp,
                'time_span_actual': (timestamp - self.timestamps[0]).total_seconds() if self.timestamps else 0,
                'time_span_target': self.time_window.total_seconds(),
                'is_full_window': (timestamp - self.timestamps[0]) >= self.time_window if self.timestamps else False,
                'ma_type': 'Time-EMA',
                'alpha_used': time_weight,
                'base_alpha': self.alpha
        }
        return self.ema1
        
    def _calculate_time_dema(self, timestamp, price, return_full_window=False):
        """Calculate time-based Double Exponential Moving Average."""
        if not self.initialized:
            # Initialize with first price
            self.ema1 = price
            self.ema2 = price
            self.initialized = True
            self.last_timestamp = timestamp
            time_weight = self.alpha
        else:
            # Calculate time-adjusted alpha
            time_weight = self._calculate_time_weight(timestamp, self.last_timestamp)
            # First EMA
            self.ema1 = time_weight * price + (1 - time_weight) * self.ema1
            # Second EMA (EMA of first EMA)
            self.ema2 = time_weight * self.ema1 + (1 - time_weight) * self.ema2
            self.last_timestamp = timestamp
        
        # DEMA = 2 * EMA1 - EMA2
        dema = 2 * self.ema1 - self.ema2
        
        if return_full_window:
            return {
                'timestamp': timestamp,
                'price': price,
                'moving_average': dema,
                'data_points_count': self.data_count,
                'time_window': self.time_window_str,
                'window_start': self.timestamps[0] if self.timestamps else timestamp,
                'window_end': timestamp,
                'time_span_actual': (timestamp - self.timestamps[0]).total_seconds() if self.timestamps else 0,
                'time_span_target': self.time_window.total_seconds(),
                'is_full_window': (timestamp - self.timestamps[0]) >= self.time_window if self.timestamps else False,
                'ma_type': 'Time-DEMA',
                'alpha_used': time_weight,
                'base_alpha': self.alpha,
                'ema1': self.ema1,
                'ema2': self.ema2
            }
        return dema
    
    def _calculate_time_tema(self, timestamp, price, return_full_window=False):
        """Calculate time-based Triple Exponential Moving Average."""
        if not self.initialized:
            # Initialize with first price
            self.ema1 = price
            self.ema2 = price
            self.ema3 = price
            self.initialized = True
            self.last_timestamp = timestamp
            time_weight = self.alpha
        else:
            # Calculate time-adjusted alpha
            time_weight = self._calculate_time_weight(timestamp, self.last_timestamp)
            # First EMA
            self.ema1 = time_weight * price + (1 - time_weight) * self.ema1
            # Second EMA (EMA of first EMA)
            self.ema2 = time_weight * self.ema1 + (1 - time_weight) * self.ema2
            # Third EMA (EMA of second EMA)
            self.ema3 = time_weight * self.ema2 + (1 - time_weight) * self.ema3
            self.last_timestamp = timestamp
        
        # TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        tema = 3 * self.ema1 - 3 * self.ema2 + self.ema3
        
        if return_full_window:
            return {
                'timestamp': timestamp,
                'price': price,
                'moving_average': tema,
                'data_points_count': self.data_count,
                'time_window': self.time_window_str,
                'window_start': self.timestamps[0] if self.timestamps else timestamp,
                'window_end': timestamp,
                'time_span_actual': (timestamp - self.timestamps[0]).total_seconds() if self.timestamps else 0,
                'time_span_target': self.time_window.total_seconds(),
                'is_full_window': (timestamp - self.timestamps[0]) >= self.time_window if self.timestamps else False,
                'ma_type': 'Time-TEMA',
                'alpha_used': time_weight,
                'base_alpha': self.alpha,
                'ema1': self.ema1,
                'ema2': self.ema2,
                'ema3': self.ema3
            }
        return tema
    
    def get_current_ma(self):
        """Get the current moving average without adding new data."""
        if self.ma_type == 'SMA':
            if len(self.data_points) == 0:
                return None
            total_price = sum(p for t, p in self.data_points)
            return total_price / len(self.data_points)
        else:
            if not self.initialized:
                return None
            if self.ma_type == 'EMA':
                return self.ema1
            elif self.ma_type == 'DEMA':
                return 2 * self.ema1 - self.ema2
            elif self.ma_type == 'TEMA':
                return 3 * self.ema1 - 3 * self.ema2 + self.ema3
    
    def get_time_window_info(self):
        """Get information about the current time window state."""
        current_time = self.timestamps[-1] if self.timestamps else None
        oldest_time = self.timestamps[0] if self.timestamps else None
        
        base_info = {
            'ma_type': f'Time-{self.ma_type}',
            'time_window_spec': self.time_window_str,
            'time_window_seconds': self.time_window.total_seconds(),
            'data_points_count': len(self.data_points),
            'total_data_processed': self.data_count,
            'current_ma': self.get_current_ma(),
            'oldest_timestamp': oldest_time,
            'newest_timestamp': current_time,
            'actual_time_span': (current_time - oldest_time).total_seconds() if current_time and oldest_time else 0,
            'window_utilization': ((current_time - oldest_time).total_seconds() / self.time_window.total_seconds() * 100) if current_time and oldest_time else 0
        }
        
        if self.ma_type != 'SMA':
            base_info.update({
                'base_alpha': self.alpha,
                'initialized': self.initialized,
                'last_calculation_time': self.last_timestamp
            })
            
            if self.initialized:
                if self.ma_type in ['EMA', 'DEMA', 'TEMA']:
                    base_info['ema1'] = self.ema1
                if self.ma_type in ['DEMA', 'TEMA']:
                    base_info['ema2'] = self.ema2
                if self.ma_type == 'TEMA':
                    base_info['ema3'] = self.ema3
        
        return base_info
    
    def reset(self):
        """Reset the moving average calculator."""
        self.data_count = 0
        self.data_points.clear()
        self.timestamps.clear()
        
        if self.ma_type != 'SMA':
            self.ema1 = None
            self.ema2 = None
            self.ema3 = None
            self.initialized = False
            self.last_timestamp = None


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


def say_hello():
    print("Hello")
    messagebox.showinfo("Greeting", "Hello")
    engine = pyttsx3.init()
    engine.say("Hello")
    engine.runAndWait()


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

# Before streaming, get the historical data for that instrument from oanda
historical_data = get_oanda_data(
    credentials=credentials,
    instrument=instrument,
    granularity='S5',  # 5-second granularity
    hours=8,  # Fetch 1 hour of historical data
    rows=rows  # Fetch up to 5000 rows of historical data
)

historical_df = pd.DataFrame(historical_data)
historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
historical_df = historical_df.drop_duplicates(subset=['timestamp'], keep='last')
# historical_df.to_csv('historical_data.csv', index=False)


# for each row in historical_df, process it with the Algo instance
for _, row in historical_df.iterrows():
    timestamp = row['timestamp']
    price = round(row['price'], precision)
    
    # Process the historical data row with the Algo instance
    take, return_dict = purple.process_row(timestamp, price, precision, say=False)
    
    # Update the live graph with historical data
    graph_updater.update_graph(return_dict)


# Start the trading script in a separate thread
trading_thread = threading.Thread(target=run_trading_script, args=(credentials,), daemon=True)
trading_thread.start()

# Run the GUI in the main thread
root = tk.Tk()
root.title("Hello GUI")

# Add a button to the GUI
hello_button = tk.Button(root, text="Say Hello", command=say_hello)
hello_button.pack(pady=20)

# Add the toggle button for 'Take'
take_button = tk.Button(root, text="Take: OFF", command=toggle_take)
take_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()

