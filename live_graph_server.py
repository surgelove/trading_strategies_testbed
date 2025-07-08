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
    'emas': deque(maxlen=maxlen),
    'temas': deque(maxlen=maxlen),
    'cross_ups': deque(maxlen=maxlen),
    'cross_downs': deque(maxlen=maxlen),
    'mamplitudes': deque(maxlen=maxlen),
    'pamplitudes': deque(maxlen=maxlen),
    'min_prices': deque(maxlen=maxlen),
    'max_prices': deque(maxlen=maxlen),
    'xmin_prices': deque(maxlen=maxlen),
    'xmax_prices': deque(maxlen=maxlen)
}

class LiveGraphUpdater:
    def __init__(self):
        self.latest_data = {}
        
    def update_graph(self, return_dict):
        """Update the live graph with new data"""
        try:
            timestamp = return_dict['timestamp']
            price = return_dict['price']
            ema = return_dict['EMA']
            tema = return_dict['TEMA']
            mamplitude = return_dict['MAmplitude']
            pamplitude = return_dict['PAmplitude']
            cross_direction = return_dict['Cross_Direction']
            cross_price_up = return_dict['Cross_Price_Up']
            cross_price_down = return_dict['Cross_Price_Down']
            min_price = return_dict['Min_Price']
            max_price = return_dict['Max_Price']
            xmin_price = return_dict['XMin_Price']  # Add XMin_Price
            xmax_price = return_dict['XMax_Price']  # Add XMax_Price

            # Add data to collections
            live_data['timestamps'].append(timestamp)
            live_data['prices'].append(price)
            live_data['emas'].append(ema)
            live_data['temas'].append(tema)
            live_data['mamplitudes'].append(mamplitude)
            live_data['pamplitudes'].append(pamplitude)
            live_data['min_prices'].append(min_price)
            live_data['max_prices'].append(max_price)
            live_data['xmin_prices'].append(xmin_price)  # Add XMin_Price to live data
            live_data['xmax_prices'].append(xmax_price)  # Add XMax_P
            
            # Handle cross points
            live_data['cross_ups'].append(cross_price_up)
            live_data['cross_downs'].append(cross_price_down)
            
            # Store latest data for API endpoint
            self.latest_data = {
                'timestamp': timestamp.strftime('%H:%M:%S'),
                'price': price,
                'ema': ema,
                'tema': tema,
                'mamplitude': mamplitude,
                'pamplitude': pamplitude,
                'cross_direction': cross_direction,
                'min_price': min_price,
                'max_price': max_price,
                'xmin_price': xmin_price,  # Add XMin_Price
                'xmax_price': xmax_price,  # Add XMax_Price
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
                emas_list = list(live_data['emas'])
                temas_list = list(live_data['temas'])
                cross_ups_list = list(live_data['cross_ups'])
                cross_downs_list = list(live_data['cross_downs'])
                min_prices_list = list(live_data['min_prices'])
                max_prices_list = list(live_data['max_prices'])
                xmin_prices_list = list(live_data['xmin_prices'])
                xmax_prices_list = list(live_data['xmax_prices'])
                mamplitudes_list = list(live_data['mamplitudes'])
                pamplitudes_list = list(live_data['pamplitudes'])
                
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
                emas_filtered = emas_list[start_idx:end_idx]
                temas_filtered = temas_list[start_idx:end_idx]
                cross_ups_filtered = cross_ups_list[start_idx:end_idx]
                cross_downs_filtered = cross_downs_list[start_idx:end_idx]
                mamplitudes_filtered = mamplitudes_list[start_idx:end_idx]
                pamplitudes_filtered = pamplitudes_list[start_idx:end_idx]
                min_prices_filtered = min_prices_list[start_idx:end_idx]
                max_prices_filtered = max_prices_list[start_idx:end_idx]
                xmin_prices_filtered = xmin_prices_list[start_idx:end_idx]
                xmax_prices_filtered = xmax_prices_list[start_idx:end_idx]
            else:
                # Use all data
                timestamps_filtered = list(live_data['timestamps'])
                prices_filtered = list(live_data['prices'])
                emas_filtered = list(live_data['emas'])
                temas_filtered = list(live_data['temas'])
                cross_ups_filtered = list(live_data['cross_ups'])
                cross_downs_filtered = list(live_data['cross_downs'])
                mamplitudes_filtered = list(live_data['mamplitudes'])
                pamplitudes_filtered = list(live_data['pamplitudes'])
                min_prices_filtered = list(live_data['min_prices'])
                max_prices_filtered = list(live_data['max_prices'])
                xmin_prices_filtered = list(live_data['xmin_prices'])
                xmax_prices_filtered = list(live_data['xmax_prices'])
            
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
            ema_trace = {
                'x': timestamps_str,
                'y': emas_filtered,
                'mode': 'lines',
                'name': 'EMA',
                'line': {'color': 'blue', 'width': 2}
            }
            
            # TEMA trace
            tema_trace = {
                'x': timestamps_str,
                'y': temas_filtered,
                'mode': 'lines',
                'name': 'TEMA',
                'line': {'color': 'purple', 'width': 2}
            }
            
            # Cross up points
            cross_up_trace = {
                'x': timestamps_str,
                'y': cross_ups_filtered,
                'mode': 'markers',
                'name': 'Cross Up',
                'marker': {'symbol': 'triangle-up', 'size': 10, 'color': 'green'}
            }
            
            # Cross down points
            cross_down_trace = {
                'x': timestamps_str,
                'y': cross_downs_filtered,
                'mode': 'markers',
                'name': 'Cross Down',
                'marker': {'symbol': 'triangle-down', 'size': 10, 'color': 'red'}
            }
            
            # Min price trace
            min_price_trace = {
                'x': timestamps_str,
                'y': min_prices_filtered,
                'mode': 'markers',
                'name': 'Min Price',
                'marker': {'symbol': 'star', 'size': 10, 'color': 'darkred', 'opacity': 1}
            }

            # Max price trace
            max_price_trace = {
                'x': timestamps_str,
                'y': max_prices_filtered,
                'mode': 'markers',
                'name': 'Max Price',
                'marker': {'symbol': 'star', 'size': 10, 'color': 'darkgreen', 'opacity': 1}
            }

            xmin_price_trace = {
                'x': timestamps_str,
                'y': xmin_prices_filtered,
                'mode': 'markers',
                'name': 'XMin Price',
                'marker': {'symbol': 'diamond', 'size': 8, 'color': 'darkred', 'opacity': 1}
            }

            # XMax price trace
            xmax_price_trace = {
                'x': timestamps_str,
                'y': xmax_prices_filtered,
                'mode': 'markers',
                'name': 'XMax Price',
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

            # MAmplitude trace
            mamplitude_trace = {
                'x': timestamps_str,
                'y': mamplitudes_filtered,
                'mode': 'lines',
                'name': 'MAmplitude',
                'line': {'color': 'orange', 'width': 1, 'opacity': 0.1},
                'yaxis': 'y2'
            }

            # PAmplitude trace
            pamplitude_trace = {
                'x': timestamps_str,
                'y': pamplitudes_filtered,
                'mode': 'lines',
                'name': 'PAmplitude',
                'line': {'color': 'lightblue', 'width': 1, 'opacity': 0.1},
                'yaxis': 'y2'
            }
            # Layout
            layout = {
                'title': f'Live Trading Data - EMA/TEMA Crossover Strategy ({len(timestamps_filtered)} points)',
                'height': 600,
                'showlegend': True,
                'hovermode': 'x unified',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Price', 'side': 'left'},
                'yaxis2': {'title': 'MAmplitude %', 'side': 'right', 'overlaying': 'y', 'range': [0, 0.11]},
                'yaxis2': {'title': 'PAmplitude %', 'side': 'right', 'overlaying': 'y', 'range': [0, 0.11]},
            }
            
            return {
                'data': [price_trace, ema_trace, tema_trace, cross_up_trace, cross_down_trace, 
                        min_price_trace, max_price_trace, xmin_price_trace, xmax_price_trace, 
                        current_price_line, pamplitude_trace, mamplitude_trace],
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
    """Run the Flask server"""
    print("🌐 Starting Flask server...")
    print("📁 Template directory:", app.template_folder)
    try:
        # Disable Flask logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # Only show errors, not request logs
        
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Server error: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    run_server()