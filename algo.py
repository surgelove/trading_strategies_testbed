

from collections import deque
import pandas as pd
from datetime import datetime, timedelta


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

        self.peak_distance = 2  # Distance in pips from the peak

        self.xtpk_distance = 2  # Distance in pips from the extreme peak
        self.xtpk_price = None
        self.xtpk_price_following = None
        self.xtpk_found = False

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
            new_travel = (price - self.peak_cross_price_previous) / self.peak_cross_price_previous * 100
            if abs(new_travel) > abs(self.peak_travel):
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
                    if abs(self.peak_travel) > self.peak_travel_threshold:       # if the price traveled enough from the last cross
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
                    if abs(self.peak_travel) > self.peak_travel_threshold:           # if the price traveled enough from the last cross
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

        # Extreme peak: when price is dramatically going in one direction, set a following price at extreme peak distance
        xtpk_cross_price = None
        if abs(self.peak_travel) > self.peak_travel_threshold:
            if self.peak_travel > 0:  # if we are going up
                ...
                # if price > self.xtpk_price:
                #     self.xtpk_price = price
                #     self.xtpk_price_following = price - self.xtpk_distance * 0.0001  # Set following price at extreme peak distance
                #     if price < self.xtpk_price_following:
                #         xtpk_cross_price = price
                #         print(f"{timestamp} XTPK: {xtpk_cross_price:.8f} Following: {self.xtpk_price_following:.8f}")
            elif self.peak_travel < 0:  # if we are going down
                if self.xtpk_price is None or price < self.xtpk_price:
                    self.xtpk_price = price
                    self.xtpk_price_following = price + self.xtpk_distance * 0.0001  # Set following price at extreme peak distance
                    print(f"Following: {self.xtpk_price_following:.8f}")
                else:
                    if self.xtpk_price_following:
                        if price > self.xtpk_price_following:
                            if not self.xtpk_found:  # Only set xtpk_cross_price if it hasn't been found yet
                                xtpk_cross_price = price
                                self.xtpk_found = True
                                print(f"{timestamp} XTPK: {xtpk_cross_price:.8f} Following: {self.xtpk_price_following:.8f}")


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
                            # if say: say_nonblocking("Cross up detected", voice="Alex")
                            base_take = 1
                        # print('up', cross_price, timestamp, tema_values[-1], ema_values[-1])  # Print timestamp and values
                        # else:
                            # if say: say_nonblocking("Cross up detected but not enough amplitude", voice="Alex")
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
                            # if say: say_nonblocking("Cross down detected", voice="Samantha")
                            base_take = -1
                        # else:
                            # if say: say_nonblocking("Cross down detected but not enough amplitude", voice="Samantha")
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
            'peak_travel': abs(self.peak_travel),
            'xtpk_cross_price': xtpk_cross_price,
            'base_min_price': self.base_min_prices[-1],
            'base_max_price': self.base_max_prices[-1],
            'aspr_min_price': self.aspr_min_prices[-1],
            'aspr_max_price': self.aspr_max_prices[-1],
            'Travel': base_travel,
            # 'Take': take
        }

        return base_take, return_dict
        

