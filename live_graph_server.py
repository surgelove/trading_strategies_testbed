from flask import Flask, render_template, jsonify, request
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import queue
import traceback

app = Flask(__name__)

# Store the latest data points (keep last 1000 points)
maxlen = 1000000
live_data = {
    'timestamps': deque(maxlen=maxlen),
    'prices': deque(maxlen=maxlen),
    'base_emas': deque(maxlen=maxlen),
    'base_temas': deque(maxlen=maxlen),
    'base_cross_price_ups': deque(maxlen=maxlen),
    'base_cross_price_downs': deque(maxlen=maxlen),
    'peak_cross_price_ups': deque(maxlen=maxlen),  # Add this
    'peak_cross_price_downs': deque(maxlen=maxlen),  # Add this
    'peak_travels': deque(maxlen=maxlen),  # Add this
    'xtpk_cross_price_ups': deque(maxlen=maxlen),  # Add xtpk_cross_price_up
    'xtpk_movements': deque(maxlen=maxlen),  # Add xtpk_movement
    'xtpk_price_dn_followings': deque(maxlen=maxlen),  # Add xtpk_price_dn_following
    'base_mamplitudes': deque(maxlen=maxlen),
    'base_pamplitudes': deque(maxlen=maxlen),
    'base_min_prices': deque(maxlen=maxlen),
    'base_max_prices': deque(maxlen=maxlen),
    'aspr_min_prices': deque(maxlen=maxlen),
    'aspr_max_prices': deque(maxlen=maxlen)
}

class LiveGraphUpdater:
    def __init__(self):
        self.latest_data = {}
        
    def update_graph(self, return_dict):
        """Update the live graph with new data"""
        try:
            timestamp = return_dict['timestamp']
            price = return_dict['price']
            base_ema = return_dict['base_ema']
            base_tema = return_dict['base_tema']
            base_mamplitude = return_dict['base_mamplitude']
            base_pamplitude = return_dict['base_pamplitude']
            base_cross_direction = return_dict['base_cross_direction']
            base_cross_price_up = return_dict['base_cross_price_up']
            base_cross_price_down = return_dict['base_cross_price_down']
            peak_cross_direction = return_dict['peak_cross_direction']  # Add this
            peak_cross_price_up = return_dict['peak_cross_price_up']  # Add this
            peak_cross_price_down = return_dict['peak_cross_price_dn']  # Add this
            peak_travel = return_dict['peak_travel']  # Add this
            xtpk_cross_price_up = return_dict['xtpk_cross_price_up']  # Add xtpk_cross_price_up
            xtpk_movement = return_dict['xtpk_movement']  # Add xtpk_movement
            xtpk_price_dn_following = return_dict['xtpk_price_dn_following']  # Add xtpk_price_dn_following
            base_min_price = return_dict['base_min_price']
            base_max_price = return_dict['base_max_price']
            aspr_min_price = return_dict['aspr_min_price']  # Add aspr_min_price
            aspr_max_price = return_dict['aspr_max_price']  # Add aspr_max_price

            # Add data to collections
            live_data['timestamps'].append(timestamp)
            live_data['prices'].append(price)
            live_data['base_emas'].append(base_ema)
            live_data['base_temas'].append(base_tema)
            live_data['base_mamplitudes'].append(base_mamplitude)
            live_data['base_pamplitudes'].append(base_pamplitude)
            live_data['peak_travels'].append(peak_travel)  # Add this
            live_data['xtpk_cross_price_ups'].append(xtpk_cross_price_up)  # Add xtpk_cross_price_up
            live_data['xtpk_movements'].append(xtpk_movement)  # Add xtpk_movement
            live_data['xtpk_price_dn_followings'].append(xtpk_price_dn_following)  # Add xtpk_price_dn_following
            live_data['base_min_prices'].append(base_min_price)
            live_data['base_max_prices'].append(base_max_price)
            live_data['aspr_min_prices'].append(aspr_min_price)  # Add aspr_min_price to live data
            live_data['aspr_max_prices'].append(aspr_max_price)  # Add aspr_max_P
            
            # Handle cross points
            live_data['base_cross_price_ups'].append(base_cross_price_up)
            live_data['base_cross_price_downs'].append(base_cross_price_down)

            # Handle peak cross points - Add these lines
            live_data['peak_cross_price_ups'].append(peak_cross_price_up)
            live_data['peak_cross_price_downs'].append(peak_cross_price_down)

            # Store latest data for API endpoint
            self.latest_data = {
                'timestamp': timestamp.strftime('%H:%M:%S'),
                'price': price,
                'base_ema': base_ema,
                'base_tema': base_tema,
                'base_mamplitude': base_mamplitude,
                'base_pamplitude': base_pamplitude,
                'peak_travel': peak_travel,
                'xtpk_movement': xtpk_movement,  # Add xtpk_movement
                'base_cross_direction': base_cross_direction,
                'peak_cross_direction': peak_cross_direction,
                'base_min_price': base_min_price,
                'base_max_price': base_max_price,
                'aspr_min_price': aspr_min_price,
                'aspr_max_price': aspr_max_price,
                'data_points': len(live_data['timestamps'])
            }
                    
        except Exception as e:
            print(f"❌ Error in update_graph: {e}")
            print(traceback.format_exc())
    
    def create_plot_data(self, start_minutes=None, end_minutes=None):
        """Create the plotly figure data with optional time range filtering"""
        try:
            if not live_data['timestamps']:
                return None
            
            # Filter data based on time range if specified
            if start_minutes is not None or end_minutes is not None:
                now = datetime.now()
                
                # Convert deques to lists for filtering
                timestamps_list = list(live_data['timestamps'])
                prices_list = list(live_data['prices'])
                base_emas_list = list(live_data['base_emas'])
                base_temas_list = list(live_data['base_temas'])
                base_cross_price_ups_list = list(live_data['base_cross_price_ups'])
                base_cross_price_downs_list = list(live_data['base_cross_price_downs'])
                peak_cross_price_ups_list = list(live_data['peak_cross_price_ups'])  # Add this
                peak_cross_price_downs_list = list(live_data['peak_cross_price_downs'])  # Add this
                peak_travels_list = list(live_data['peak_travels'])  # Add this
                xtpk_cross_price_ups_list = list(live_data['xtpk_cross_price_ups'])  # Add xtpk_cross_price_up
                xtpk_movements_list = list(live_data['xtpk_movements'])  # Add xtpk_movement
                xtpk_price_dn_followings_list = list(live_data['xtpk_price_dn_followings'])  # Add xtpk_price_dn_following
                base_min_prices_list = list(live_data['base_min_prices'])
                base_max_prices_list = list(live_data['base_max_prices'])
                aspr_min_prices_list = list(live_data['aspr_min_prices'])
                aspr_max_prices_list = list(live_data['aspr_max_prices'])
                base_mamplitudes_list = list(live_data['base_mamplitudes'])
                base_pamplitudes_list = list(live_data['base_pamplitudes'])
                
                # Find start and end indices
                start_idx = 0
                end_idx = len(timestamps_list)
                
                if start_minutes is not None:
                    start_time = now - timedelta(minutes=start_minutes)
                    for i, ts in enumerate(timestamps_list):
                        if ts >= start_time:
                            start_idx = i
                            break
                
                if end_minutes is not None:
                    end_time = now - timedelta(minutes=end_minutes)
                    for i, ts in enumerate(timestamps_list):
                        if ts >= end_time:
                            end_idx = i
                            break
                
                # Slice the data
                timestamps_filtered = timestamps_list[start_idx:end_idx]
                prices_filtered = prices_list[start_idx:end_idx]
                base_emas_filtered = base_emas_list[start_idx:end_idx]
                base_temas_filtered = base_temas_list[start_idx:end_idx]
                base_cross_ups_filtered = base_cross_price_ups_list[start_idx:end_idx]
                base_cross_downs_filtered = base_cross_price_downs_list[start_idx:end_idx]
                peak_cross_ups_filtered = peak_cross_price_ups_list[start_idx:end_idx]  # Add this
                peak_cross_downs_filtered = peak_cross_price_downs_list[start_idx:end_idx]  # Add this
                peak_travels_filtered = peak_travels_list[start_idx:end_idx]  # Add this
                xtpk_cross_price_ups_filtered = xtpk_cross_price_ups_list[start_idx:end_idx]  # Add xtpk_cross_price_up
                xtpk_movements_filtered = xtpk_movements_list[start_idx:end_idx]  # Add xtpk_movement
                xtpk_price_dn_followings_filtered = xtpk_price_dn_followings_list[start_idx:end_idx]  # Add xtpk_price_dn_following
                base_mamplitudes_filtered = base_mamplitudes_list[start_idx:end_idx]
                base_pamplitudes_filtered = base_pamplitudes_list[start_idx:end_idx]
                base_min_prices_filtered = base_min_prices_list[start_idx:end_idx]
                base_max_prices_filtered = base_max_prices_list[start_idx:end_idx]
                aspr_min_prices_filtered = aspr_min_prices_list[start_idx:end_idx]
                aspr_max_prices_filtered = aspr_max_prices_list[start_idx:end_idx]
            else:
                # Use all data
                timestamps_filtered = list(live_data['timestamps'])
                prices_filtered = list(live_data['prices'])
                base_emas_filtered = list(live_data['base_emas'])
                base_temas_filtered = list(live_data['base_temas'])
                base_cross_ups_filtered = list(live_data['base_cross_price_ups'])
                base_cross_downs_filtered = list(live_data['base_cross_price_downs'])
                peak_cross_ups_filtered = list(live_data['peak_cross_price_ups'])  # Add this
                peak_cross_downs_filtered = list(live_data['peak_cross_price_downs'])  # Add this
                peak_travels_filtered = list(live_data['peak_travels'])  # Add this
                xtpk_cross_price_ups_filtered = list(live_data['xtpk_cross_price_ups'])  # Add xtpk_cross_price_up
                xtpk_movements_filtered = list(live_data['xtpk_movements'])  # Add xtpk_movement
                xtpk_price_dn_followings_filtered = list(live_data['xtpk_price_dn_followings'])  # Add xtpk_price_dn_following
                base_mamplitudes_filtered = list(live_data['base_mamplitudes'])
                base_pamplitudes_filtered = list(live_data['base_pamplitudes'])
                base_min_prices_filtered = list(live_data['base_min_prices'])
                base_max_prices_filtered = list(live_data['base_max_prices'])
                aspr_min_prices_filtered = list(live_data['aspr_min_prices'])
                aspr_max_prices_filtered = list(live_data['aspr_max_prices'])
            
            if not timestamps_filtered:
                return None
                
            # Convert timestamps to strings for JSON serialization
            timestamps_str = [t.strftime('%H:%M:%S') for t in timestamps_filtered]
            
            # Main price chart
            price_trace = {
                'x': timestamps_str,
                'y': prices_filtered,
                'mode': 'lines+markers',
                'name': 'Price',
                'line': {'color': 'gray', 'width': 1, 'opacity': 0.5},
                'marker': {'size': 3}
            }
            
            # EMA trace
            base_ema_trace = {
                'x': timestamps_str,
                'y': base_emas_filtered,
                'mode': 'lines',
                'name': 'Base EMA',
                'line': {'color': 'blue', 'width': 2}
            }
            
            # TEMA trace
            base_tema_trace = {
                'x': timestamps_str,
                'y': base_temas_filtered,
                'mode': 'lines',
                'name': 'Base TEMA',
                'line': {'color': 'purple', 'width': 2}
            }
            
            # Cross up points
            base_cross_up_trace = {
                'x': timestamps_str,
                'y': base_cross_ups_filtered,
                'mode': 'markers',
                'name': 'Base Cross Up',
                'marker': {'symbol': 'triangle-up', 'size': 10, 'color': 'green'}
            }
            
            # Cross down points
            base_cross_down_trace = {
                'x': timestamps_str,
                'y': base_cross_downs_filtered,
                'mode': 'markers',
                'name': 'Base Cross Down',
                'marker': {'symbol': 'triangle-down', 'size': 10, 'color': 'red'}
            }

            # Peak cross up points - Add this
            peak_cross_up_trace = {
                'x': timestamps_str,
                'y': peak_cross_ups_filtered,
                'mode': 'markers',
                'name': 'Peak Cross Up',
                'marker': {'symbol': 'triangle-up', 'size': 8, 'color': 'lightgreen', 'line': {'color': 'green', 'width': 2}}
            }
            
            # Peak cross down points - Add this
            peak_cross_down_trace = {
                'x': timestamps_str,
                'y': peak_cross_downs_filtered,
                'mode': 'markers',
                'name': 'Peak Cross Down',
                'marker': {'symbol': 'triangle-down', 'size': 8, 'color': 'lightcoral', 'line': {'color': 'red', 'width': 2}}
            }

            # Peak Travel trace - Add this
            peak_travel_trace = {
                'x': timestamps_str,
                'y': peak_travels_filtered,
                'mode': 'lines',
                'name': 'Peak Travel',
                'line': {'color': 'orange', 'width': 2},
                'yaxis': 'y2'  # Use second y-axis for travel percentage
            }
            
            # Min price trace
            base_min_price_trace = {
                'x': timestamps_str,
                'y': base_min_prices_filtered,
                'mode': 'markers',
                'name': 'Base Min Price',
                'marker': {'symbol': 'star', 'size': 10, 'color': 'darkred', 'opacity': 1}
            }

            # Max price trace
            base_max_price_trace = {
                'x': timestamps_str,
                'y': base_max_prices_filtered,
                'mode': 'markers',
                'name': 'Base Max Price',
                'marker': {'symbol': 'star', 'size': 10, 'color': 'darkgreen', 'opacity': 1}
            }

            aspr_min_price_trace = {
                'x': timestamps_str,
                'y': aspr_min_prices_filtered,
                'mode': 'markers',
                'name': 'Aspr Min Price',
                'marker': {'symbol': 'diamond', 'size': 8, 'color': 'darkred', 'opacity': 1}
            }

            # aspr_max price trace
            aspr_max_price_trace = {
                'x': timestamps_str,
                'y': aspr_max_prices_filtered,
                'mode': 'markers',
                'name': 'Aspr Max Price',
                'marker': {'symbol': 'diamond', 'size': 8, 'color': 'darkgreen', 'opacity': 1}
            }

            # Current price horizontal line
            current_price = prices_filtered[-1] if prices_filtered else 0
            current_price_line = {
                'x': [timestamps_str[0], timestamps_str[-1]] if timestamps_str else [],
                'y': [current_price, current_price],
                'mode': 'lines',
                'name': 'Current Price Line',
                'line': {'color': 'gray', 'width': 1, 'dash': 'dash'},
                'hovertemplate': f'Current Price: {current_price:.5f}<extra></extra>'
            }

            # base_mamplitude trace
            base_mamplitude_trace = {
                'x': timestamps_str,
                'y': base_mamplitudes_filtered,
                'mode': 'lines',
                'name': 'Base Mamplitude',
                'line': {'color': 'orange', 'width': 1, 'opacity': 0.1},
                'yaxis': 'y2'
            }

            # base_pamplitude trace
            base_pamplitude_trace = {
                'x': timestamps_str,
                'y': base_pamplitudes_filtered,
                'mode': 'lines',
                'name': 'Base Pamplitude',
                'line': {'color': 'lightblue', 'width': 1, 'opacity': 0.1},
                'yaxis': 'y2'
            }
            # XTPK cross points - Add this new trace
            xtpk_cross_price_up_trace = {
                'x': timestamps_str,
                'y': xtpk_cross_price_ups_filtered,
                'mode': 'markers',
                'name': 'XTPK Cross Price Up',
                'marker': {'symbol': 'triangle-up', 'size': 8, 'color': 'lightgreen', 'line': {'color': 'green', 'width': 2}}
            }

            # XTPK Movement trace - Add this new trace
            xtpk_movement_trace = {
                'x': timestamps_str,
                'y': xtpk_movements_filtered,
                'mode': 'lines',
                'name': 'XTPK Movement',
                'line': {'color': 'lightgreen', 'width': 2},
                'yaxis': 'y2'  # Use second y-axis for movement percentage
            }

            # XTPK Price Following trace - Add this new trace
            xtpk_price_dn_following_trace = {
                'x': timestamps_str,
                'y': xtpk_price_dn_followings_filtered,
                'mode': 'lines',
                'name': 'XTPK Price Following Down',
                'line': {'color': 'purple', 'width': 3}
            }

            # # TEMA trace
            # base_tema_trace = {
            #     'x': timestamps_str,
            #     'y': base_temas_filtered,
            #     'mode': 'lines',
            #     'name': 'Base TEMA',
            #     'line': {'color': 'purple', 'width': 2}
            # }

            # Layout
            layout = {
                'title': f'Live Trading Data - EMA/TEMA Crossover Strategy ({len(timestamps_filtered)} points)',
                'height': 600,
                'showlegend': True,
                'hovermode': 'x unified',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Price', 'side': 'left'},
                'yaxis2': {'title': 'base_mamplitude %', 'side': 'right', 'overlaying': 'y'},
            }
            
            return {
                'data': [price_trace, base_ema_trace, base_tema_trace, base_cross_up_trace, base_cross_down_trace, 
                        peak_cross_up_trace, peak_cross_down_trace, peak_travel_trace, xtpk_cross_price_up_trace,
                        xtpk_movement_trace, xtpk_price_dn_following_trace, base_min_price_trace, base_max_price_trace, 
                        aspr_min_price_trace, aspr_max_price_trace, current_price_line, base_pamplitude_trace, base_mamplitude_trace],
                'layout': layout
            }
            
        except Exception as e:
            print(f"❌ Error in create_plot_data: {e}")
            print(traceback.format_exc())
            return None

# Global updater instance
graph_updater = LiveGraphUpdater()

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('live_graph_simple.html')
    except Exception as e:
        print(f"❌ Error serving template: {e}")
        print(traceback.format_exc())
        return f"<h1>Template Error</h1><p>Error: {e}</p><p>Make sure templates/live_graph_simple.html exists</p>", 500

@app.route('/data')
def get_current_data():
    """API endpoint to get current data"""
    try:
        if graph_updater.latest_data:
            return jsonify(graph_updater.latest_data)
        return jsonify({'message': 'No data available'})
    except Exception as e:
        print(f"❌ Error in /data endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/plot_data')
def get_plot_data():
    """API endpoint to get plot data with optional time range"""
    try:
        # Get time range parameters from query string
        start_minutes = request.args.get('start_minutes', type=int)
        end_minutes = request.args.get('end_minutes', type=int)
        
        plot_data = graph_updater.create_plot_data(start_minutes, end_minutes)
        if plot_data:
            return jsonify(plot_data)
        return jsonify({'message': 'No plot data available'})
    except Exception as e:
        print(f"❌ Error in /plot_data endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    """API endpoint to get server status"""
    return jsonify({
        'status': 'running',
        'data_points': len(live_data['timestamps']),
        'latest_timestamp': live_data['timestamps'][-1].strftime('%H:%M:%S') if live_data['timestamps'] else None,
        'oldest_timestamp': live_data['timestamps'][0].strftime('%H:%M:%S') if live_data['timestamps'] else None,
        'time_span_minutes': ((live_data['timestamps'][-1] - live_data['timestamps'][0]).total_seconds() / 60) if len(live_data['timestamps']) > 1 else 0
    })

def run_server():
    """Run the Flask server"""
    print("🌐 Starting Flask server...")
    try:
        # Disable Flask logging to reduce startup overhead
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # Only show errors, not request logs
        
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Server error: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    run_server()