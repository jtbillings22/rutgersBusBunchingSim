import numpy as np
import pandas as pd
import os 
from scipy import stats
import matplotlib.pyplot as plt
import busUtils as utils

# read bus data
busbreaks = pd.read_csv(os.path.join('rutgers_bus_data','bus_breaks.csv'))
routes = pd.read_csv(os.path.join('rutgers_bus_data','routes.csv'))
vehicles = pd.read_csv(os.path.join('rutgers_bus_data','vehicles.csv'),dtype={"routeName": str})

# filtering bus data
vehicles['routeName'] = vehicles['routeName'].str.replace(' Route', '', regex=False)
busbreaks['stop_id'] = busbreaks['stop_id'].astype(str).str.strip().str.upper()
routes['stop_sequence'] = routes['stop_sequence'].astype(str).str.strip().str.upper()
routes['route_id'] = routes['route_id'].astype(str).str.strip().str.upper()
vehicles['routeName'] = vehicles['routeName'].astype(str).str.strip().str.upper()

# removes unneccessary routes
routes = routes[routes["route_id"].isin(vehicles["routeName"])]



# merging data
busbreaks = busbreaks.merge(vehicles[['id', 'routeName']], on='id', how='left')
#print(busbreaks.head)


# generate all the pdf's for the buses.
buses = {}
for _, row in routes.iterrows():
    route_id = row['route_id']
    curr_data = busbreaks[busbreaks["routeName"] == route_id].copy()

    # quick filtering step to prevent data-type mismatching
    curr_data["stop_id"] = curr_data["stop_id"].astype(int)

    # so curr_data is the mini busbreaks data corresponding to route_id
    # now we want to iterate for each stop, creating another mini dataframe and computing the pdf on that.
    # route_id also has a stop_seq, this is what we need next
    stop_seq = [int(s) for s in row["stop_sequence"].split(",")]
    #print('current route: ', route_id)
    
    bus = utils.busRoute(route_id)
    buses[route_id] = bus # adds the bus to our bus dictionary where the route_id is the index
    print('current route ', route_id )
    for stop_id in stop_seq:
        stop_data = curr_data[curr_data['stop_id'] == stop_id]
        stop_data = stop_data.reset_index(drop=True) # reset the indices so it looks nicer :)

        #print('current stop ', stop_id, 'break duration vector: ', stop_data['break_duration'])
        # so now at this step, we want to create the bus object with it's given pdf
        # right now, we have access to the stop_data dataset, as well as the route_id, and stop_id.
        # what we will do is calculate the pdf for the given stop_id, using busRoute.genPDF(stop_data, stop_id)
        bus.generate_stop_pdf(stop_id, stop_data['break_duration']) # only send in the break_duration vector of the dataframe
        bus.generate_pdf_image(stop_id)


# take a few samples from the LX bus
        
        

#df_x = busbreaks[busbreaks["routeName"] == "REXL"]
#print(df_x)

# okay so each bus has a "routeName", and then there is an associated bus id number, saved as "id".
# we want the data of every stop each bus makes. so for each unique id, create a mini sample of it's respective id, then
# compute it's pdf.

