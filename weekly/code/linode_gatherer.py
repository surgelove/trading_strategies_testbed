import requests
import json
import statistics
import datetime
import time
import sqlite3

api_hostname = 'api-fxpractice.oanda.com'
stream_hostname = 'stream-fxpractice.oanda.com'
api_token = 'ea1b82310c6df664e2a35d1006595f01-29b534e01de5707ce161ba8e7a097aa3'
account_id = '101-002-12713542-007'

OANDA_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
OANDA_LEN = 26
MASTER_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

db_path = 'db/'

# Open a connection to the big db, create it if it does not exist
con = sqlite3.connect(db_path + 'aia_big.db')
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS gather(date, instrument, buy, sell)")

# Open a connection to the small db, create it if it does not exist
con_sml = sqlite3.connect(db_path + 'aia.db')
cur_sml = con_sml.cursor()
cur_sml.execute("CREATE TABLE IF NOT EXISTS gather(date, instrument, buy, sell)")

# Create mapping table in small db if not exists
cur_sml.execute("CREATE TABLE IF NOT EXISTS instrument(id, instrument_name)")
cur_sml.execute("CREATE UNIQUE INDEX IF NOT EXISTS udx_instrument ON instrument(instrument_name)")

# Get instruments list
bearer = f"Bearer {api_token}"
headers = {'Authorization': bearer}
response = requests.get(f'https://{api_hostname}/v3/accounts/{account_id}/instruments', headers=headers)
data = json.loads(response.content)

# Populate instruments list and instrument table
instrument_list = ''
for i, instrument in enumerate(data['instruments']):

    # Add to mapping table if not there
    instrument_name = instrument['name']
    try:
        cur_sml.execute(f"INSERT INTO instrument VALUES ({i+1}, '{instrument_name}')")
        con_sml.commit()
    except Exception as e:
        con_sml.rollback()

    # Add to list of instruments that the gatherer needs to gather
    instrument_list += f"{instrument_name},"


def gather():

    # Build dict based on instrument table
    exe = cur_sml.execute(f"SELECT id, instrument_name FROM instrument;")
    res = exe.fetchall()
    instrument_mapping = dict((y, x) for x, y in res)

    params = {'instruments': instrument_list}
    bearer = f"Bearer {api_token}"
    headers = {'Authorization': bearer}
    response = requests.get(f'https://{stream_hostname}/v3/accounts/{account_id}/pricing/stream', params=params, headers=headers,stream=True)

    started = False
    counter = 0
    print_every = 1000

    inst_lst = []

    try:
        with response:
            if not started:
                print('👍', end='', flush=True)
                started = True
            for line in response.iter_lines(decode_unicode=True):
                data = json.loads(line)
                if data['type'] == 'PRICE':

                    timestamp = datetime.datetime.strptime(
                        data['time'][:OANDA_LEN],
                        OANDA_FORMAT
                    ).replace(tzinfo=datetime.timezone.utc).astimezone()

                    date = timestamp.strftime(MASTER_FORMAT)
                    small_date = timestamp.timestamp()
                    instrument = data['instrument']
                    instrument_id = instrument_mapping[instrument]
                    buy = statistics.mean([float(a['price']) for a in data['asks']])
                    sell = statistics.mean([float(b['price']) for b in data['bids']])

                    instrument_found = False
                    instrument_changed = False
                    for l in inst_lst:
                        
                        if l['instrument'] == instrument:
                            instrument_found = True
                            if l['buy'] == buy and l['sell'] == sell :
                                ...  # Same instrument values, do nothing
                            else:
                                l['instrument'] = instrument
                                l['buy'] = buy
                                l['sell'] = sell
                                instrument_changed = True

                    if not instrument_found:
                        dct = {}
                        dct['instrument'] = instrument
                        dct['buy'] = buy
                        dct['sell'] = sell
                        inst_lst.append(dct)
                        instrument_changed = True

                    if instrument_changed:
                        sql = f"INSERT INTO gather VALUES ('{date}', '{instrument}', {buy}, {sell})"
                        cur.execute(sql)
                        con.commit()
                        
                        sql = f"INSERT INTO gather VALUES ('{small_date}', '{instrument_id}', {buy}, {sell})"
                        cur_sml.execute(sql)
                        con_sml.commit()

                        counter += 1
                        if counter % print_every == 0:
                            print(f'{counter} ', end='', flush=True)  

                else:
                    ...  # heartbeat

    except Exception as e:
        # print(e)
        print('👎', end='', flush=True)
        time.sleep(5)
        gather()

gather()