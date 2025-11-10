from flask import Flask, render_template, request, jsonify
import busUtils as utils
import numpy as np
import pandas as pd
import os 
from scipy import stats
import matplotlib.pyplot as plt
import random
import json
import time

# before we start simulation, need to generate the pdfs for the bus routes.

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

# now we have a buses dict which stores all the neccesary information for retrieving samples from the pdfs.
# so now, we need to start the implementation of the simulation.

app = Flask(__name__)

LAST_UPDATE_TIME = time.time()
CURRENT_ROUTE_ID = "A" # global route_id
SIM_SPEED = 1.0
EVENT_LOG = []     # stores stop events
MAX_EVENTS = 100   # cap log size to avoid memory bloat
METRICS = {
    "frame_count": 0,
    "bunch_frames": 0,
    "cluster_frames": 0,
    "current": {
        "headway_score": 0.0,
        "cluster_score": 0.0,
        "bunch_score": 0.0,
        "bunch_count": 0,
        "cac_count": 0,
        "liv_count": 0
    },
    "history": []
}


# === Load routes.json ===
with open("routes(1).json") as f:
    ROUTES = json.load(f)

def normalize_route(route):
    """Compute cumulative normalized distances along a route."""
    route = np.array(route)
    if len(route) < 2:
        return route.tolist(), [0.0] * len(route)
    d = np.sqrt(np.sum(np.diff(route, axis=0)**2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    s /= s[-1]  # normalize to [0,1]
    return route.tolist(), s.tolist()

def map_stops_to_norm(path, norm, stops):
    """Assign a normalized t position to each stop based on nearest point on path."""
    stop_list = []
    for s in stops:
        sx, sy = (s["x"], s["y"]) if isinstance(s, dict) else s
        # Find nearest point on route
        dists = [((px - sx)**2 + (py - sy)**2) for px, py in path]
        idx = int(np.argmin(dists))
        stop_list.append({
            "id": s["id"] if isinstance(s, dict) and "id" in s else f"S{len(stop_list)+1}",
            "x": sx,
            "y": sy,
            "t": float(norm[idx]),  # normalized position along route
            "name": s.get("name", f"Stop {len(stop_list)+1}") if isinstance(s, dict) else f"Stop {len(stop_list)+1}"
        })
    return stop_list

def get_route(prefix):
    for key in ROUTES:
        if key.lower().startswith(prefix.lower()):
            return ROUTES[key]
    raise KeyError(f"No route found for '{prefix}'")

def init_default_sim(route_id="A"):
    """Initialize a valid simulation state on startup."""
    global CURRENT_ROUTE_ID, PATH, NORM, STOPS, runtime_buses
    if route_id not in ROUTES:
        route_id = list(ROUTES.keys())[0]  # fallback to first available route

    CURRENT_ROUTE_ID = route_id
    route_data = ROUTES[route_id]
    PATH = route_data["path"]
    NORM = route_data["norm"]
    STOPS = route_data["stops"]

    runtime_buses = []
    for i in range(NUM_BUSES):
        runtime_buses.append({
            "id": f"Bus {i+1}",
            "route_id": route_id,
            "path": PATH,
            "stops": STOPS,
            "pos": i / NUM_BUSES,
            "speed": 0.002 + random.random() * 0.002,
            "wait": 0,
            "last_stop": None,
            "model": list(buses.values())[0]
        })
    print(f"[init_default_sim] Initialized route {route_id}, {len(PATH)} points, {len(STOPS)} stops.")

# immediately normalize all the routes so we don't have to recompute them when we want to run other bus sims
# already have all routes generated so this is good, now we just want to dynamically select the route.
for key, data in ROUTES.items():
    path, norm = normalize_route(data["path"])
    stops = map_stops_to_norm(path, norm, data["stops"])
    data["path"] = path
    data["norm"] = norm
    data["stops"] = stops

route_data = ROUTES["A"] # make this dynamic stop
PATH = route_data["path"]
NORM = route_data["norm"]
STOPS = route_data["stops"]

# initialize bus states
NUM_BUSES = 4 # we want to dynamically choose this num_buses, but we can default to 4 buses.
runtime_buses = []

for route_id, bus_model in buses.items():
    route_data = get_route(route_id)
    PATH = route_data["path"]

    # normalize stops into dict list
    stops = []
    for i, s in enumerate(route_data["stops"], start=1):
        if isinstance(s, dict):
            stops.append(s)
        else:
            stops.append({"id": f"S{i}", "x": s[0], "y": s[1]})

    # create several runtime buses for this route
    for i in range(NUM_BUSES):
        runtime_buses.append({
            "id": f"Bus {i+1}",
            "route_id": route_id,
            "path": PATH,
            "stops": stops,
            "pos": i / NUM_BUSES,
            "speed": 0.002 + random.random() * 0.002,
            "wait": 0,
            "last_stop": None,
            "model": bus_model  # attach your real busRoute object here
        })

@app.route("/")
def index():
    return render_template("index.html")

# selects route and resets the buses.
@app.route("/set_route", methods=["POST"])
def set_route():
    # declare all globals 
    global CURRENT_ROUTE_ID, runtime_buses, PATH, NORM, STOPS
    
    # debugging 
    try:
        data = request.get_json(force=True)
        print("[set_route] Raw data:", request.data)
        print("[set_route] Parsed JSON:", data)
    except Exception as e:
        print("[set_route] JSON parse error:", e)
        return jsonify({"error": "Could not parse JSON"}), 400

    route_id = (data or {}).get("route_id")
    print("[set_route] route_id received:", route_id)

    if not route_id:
        return jsonify({"error": "Missing route_id"}), 400
    if route_id not in ROUTES:
        print("[set_route] Invalid route id! Available keys:", list(ROUTES.keys()))
        return jsonify({"error": f"Invalid route '{route_id}'"}), 400



    data = request.get_json(force=True)
    route_id = data.get("route_id")

    if not route_id:
        return jsonify({"error": "Missing route_id"}), 400
    if route_id not in ROUTES:
        return jsonify({"error": f"Invalid route '{route_id}'"}), 400

    CURRENT_ROUTE_ID = route_id
    route_data = ROUTES[route_id]

    # update global route data
    PATH = route_data["path"]
    NORM = route_data["norm"]
    STOPS = route_data["stops"]

    #  reset runtime buses for the new route
    runtime_buses = []
    for i in range(NUM_BUSES):
        runtime_buses.append({
            "id": f"Bus {i+1}",
            "route_id": route_id,
            "path": PATH,
            "stops": STOPS,
            "pos": i / NUM_BUSES,
            "speed": 0.002 + random.random() * 0.002,
            "wait": 0,
            "last_stop": None,
            "model": list(buses.values())[0],
            "at_stop": False,
            "dwell_time": 0,
            "last_pos": None
        })

    print(f"[set_route] Route switched to {CURRENT_ROUTE_ID}, PATH len={len(PATH)}, NORM len={len(NORM)}")
    return jsonify({"ok": True, "route": CURRENT_ROUTE_ID})


@app.route("/meta")
def meta():
    global CURRENT_ROUTE_ID
    active_route_id = CURRENT_ROUTE_ID  # Or dynamically selected later, now is when we dynamicaly select it, should be = route_id
    active_buses = [bus for bus in runtime_buses if bus["route_id"] == active_route_id]
    info = {
        "route_id": active_route_id,
        "num_buses": len(active_buses),
        "num_stops": len(active_buses[0]["stops"]) if active_buses else 0,
        "speed_factor": SIM_SPEED
    }
    return jsonify(info)

@app.route("/positions")
def positions(): 
    global EVENT_LOG, SIM_SPEED, CURRENT_ROUTE_ID, LAST_UPDATE_TIME
    active_route_id = CURRENT_ROUTE_ID
    
    # use time instead of tick updates to reflect stopping times
    now = time.time()
    dt = (now - LAST_UPDATE_TIME) * SIM_SPEED
    LAST_UPDATE_TIME = now

    active_buses = [bus for bus in runtime_buses if bus["route_id"] == active_route_id]
    
    # error catch
    if not active_buses:
        return jsonify({"error": f"No buses found for route '{active_route_id}'"})


    path = PATH
    stops = STOPS
    for bus in active_buses:
        if bus["wait"] > 0:
            bus["wait"] -= dt
            if bus["wait"] <= 0:
                bus["wait"] = 0
                bus["at_stop"] = False   # finished stop, resume moving
            continue

        # record previous position
        if bus["last_pos"] is None:
            bus["last_pos"] = interpolate(path, NORM, bus["pos"])
        x_prev, y_prev = bus["last_pos"]


        # move along path scaled by speed factor
        bus["pos"] = (bus["pos"] + bus["speed"] * dt * SIM_SPEED) % 1.0
        x, y = interpolate(path, NORM, bus["pos"])
        bus["last_pos"] = (x,y)

        # stop detection
        for stop in stops:
            sx, sy = stop["x"], stop["y"]
            stop_id = int(stop["id"])

            # use a more robust check: did the bus path between frames cross near the stop?
            
            # this needs to fix! the check_segment is bugging and snapping buses to wrong spot
            if check_segment_near_stop(x_prev, y_prev, x, y, sx, sy, radius=6):
                # prevent double-triggering while already stopped at this stop
                if bus["at_stop"] or bus.get("last_stop_id") == stop_id:
                    break

                try:
                    dwell = float(bus["model"].sampleStop(stop_id))
                    bus["wait"] = dwell
                    bus["dwell_time"] = dwell
                    bus["total_dwell"] = dwell
                    bus["at_stop"] = True
                    bus["last_stop"] = stop_id
                    bus["last_stop_id"] = stop_id

                    # log this stop event
                    stop_name = stop.get("name", f"Stop{stop_id}")
                    EVENT_LOG.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "bus_id": bus["id"],
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "dwell": round(float(dwell), 1)
                    })
                    if len(EVENT_LOG) > MAX_EVENTS:
                        EVENT_LOG = EVENT_LOG[-MAX_EVENTS:]

                    print(f"{bus['id']} reached stop {stop_id} → dwell {dwell:.1f}s")
                except Exception as e:
                    print(f"[WARN] Could not sample stop {stop_id} for route {bus['route_id']}: {e}")
                    bus["wait"] = random.randint(20, 60)
                break
        

        # reset last_stop_id if bus has moved away
        if bus.get("last_stop_id") is not None:
            last_stop = next((s for s in stops if int(s["id"]) == bus["last_stop_id"]), None)
            if last_stop:
                lx, ly = last_stop["x"], last_stop["y"]
                if ((x - lx)**2 + (y - ly)**2)**0.5 > 20:
                    bus["last_stop_id"] = None

        # small random speed variation
        bus["speed"] += random.uniform(-0.00002, 0.00002)
        bus["speed"] = max(0.002, min(0.004, bus["speed"]))


    # backend function for the dwell time progress bar
    serializable_buses = []
    for b in active_buses:
        current_stop = None
        if b["at_stop"]:
            # progress = 1 - (remaining wait / total dwell)
            total = b.get("total_dwell", b["wait"]) or 1.0
            remaining = max(b["wait"], 0.0)
            progress = 1 - (remaining / total)
            current_stop = {
                "stop_id": b["last_stop_id"],
                "stop_name": next((s["name"] for s in STOPS if int(s["id"]) == b["last_stop_id"]), f"Stop{b['last_stop_id']}"),
                "progress": round(progress, 3),
                "remaining": round(remaining, 1),
                "total": round(total, 1)
            }
        elif b.get("last_stop_id") is not None:
            # last completed stop
            current_stop = {
                "stop_id": b["last_stop_id"],
                "stop_name": next((s["name"] for s in STOPS if int(s["id"]) == b["last_stop_id"]), f"Stop{b['last_stop_id']}"),
                "progress": 1.0,
                "remaining": 0.0,
                "total": round(b.get("total_dwell", 0.0), 1)
            }

        serializable_buses.append({
            "id": b["id"],
            "route_id": b["route_id"],
            "pos": b["pos"],
            "wait": b["wait"],
            "last_stop": b["last_stop"],
            "status": current_stop
        })    


    # this needs some serious updating...
    # --- METRICS COMPUTATION BLOCK ---
    METRICS["frame_count"] += 1
    route_positions = []

    # collect positions along route for active route
    for bus in active_buses:
        route_positions.append(bus["pos"])
    route_positions = sorted(route_positions)

    # expected headway (uniform spacing)
    n = len(route_positions)
    expected_headway = 1.0 / n if n > 0 else 1.0
    tau = 0.3

    # compute actual headways (distance between consecutive buses on loop)
    headways = [ (route_positions[(i+1)%n] - route_positions[i]) % 1.0 for i in range(n) ]
    bunched = [h < tau * expected_headway for h in headways]
    bunch_count = sum(bunched)

    headway_score = bunch_count / n if n > 0 else 0.0

    # Campus clustering
    # define zone ranges (example: normalized along route)
    CAV_ZONE = (0.1, 0.4)   # CASC–SAC region (adjust for your route)
    BUSCH_ZONE = (0.6, 0.9)   # Plaza–Quads region

    cav_count = sum(CAV_ZONE[0] <= p <= CAV_ZONE[1] for p in route_positions)
    busch_count = sum(BUSCH_ZONE[0] <= p <= BUSCH_ZONE[1] for p in route_positions)
    cluster_score = 0.0
    cluster_event = False
    if n > 0 and (cav_count >= (2/3)*n or busch_count >= (2/3)*n):
        cluster_score = 1.0
        cluster_event = True

    # --- Weighted bunch score ---
    bunch_score = 0.4 * headway_score + 0.6 * cluster_score

    # --- Update cumulative ratios ---
    if bunch_count > 0:
        METRICS["bunch_frames"] += 1
    if cluster_event:
        METRICS["cluster_frames"] += 1

    # compute running ratios
    bunch_ratio = METRICS["bunch_frames"] / METRICS["frame_count"]
    cluster_ratio = METRICS["cluster_frames"] / METRICS["frame_count"]

    # store snapshot
    METRICS["current"].update({
        "headway_score": headway_score,
        "cluster_score": cluster_score,
        "bunch_score": bunch_score,
        "bunch_count": bunch_count,
        "cac_count": cav_count,
        "liv_count": busch_count,
        "bunch_ratio": bunch_ratio,
        "cluster_ratio": cluster_ratio
    })
    METRICS["history"].append(METRICS["current"].copy())
    if len(METRICS["history"]) > 1000:
        METRICS["history"].pop(0)

    print(f"[DEBUG] sending to frontend: {len(serializable_buses)} buses, {len(path)} path points")

    return jsonify({
        "buses": serializable_buses,
        "path": path,
        "norm": NORM,
        "stops": stops
    })

@app.route("/metrics")
def metrics():
    return jsonify(METRICS["current"])

@app.route("/events")
def events():
    return jsonify(EVENT_LOG[-20:])  # send last 20 messages

@app.route("/speed", methods=["POST"])
def set_speed():
    global SIM_SPEED
    try:
        factor = float(request.json.get("factor", 1.0))
        SIM_SPEED = max(0.1, min(factor, 50.0))
        print(f"Simulation speed set to {SIM_SPEED}x")
        return jsonify({"ok": True, "speed_factor": SIM_SPEED})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def interpolate(points, norm, t):
    """Interpolate position along route using normalized arc lengths."""
    if len(points) < 2:
        return points[0] if points else (0, 0)
    target = t % 1.0
    for i in range(len(norm) - 1):
        if norm[i] <= target <= norm[i + 1]:
            frac = (target - norm[i]) / (norm[i + 1] - norm[i])
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            return (x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac)
    return points[-1]

def check_segment_near_stop(x1, y1, x2, y2, sx, sy, radius=6):
    """
    Return True if the line segment (x1,y1)-(x2,y2) passes within 'radius' of (sx,sy).
    """
    # vector math: project stop onto segment and measure shortest distance
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        # degenerate case: didn't move
        return ((x1 - sx)**2 + (y1 - sy)**2)**0.5 <= radius

    t = ((sx - x1)*dx + (sy - y1)*dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))  # clamp to segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    dist = ((nearest_x - sx)**2 + (nearest_y - sy)**2)**0.5
    return dist <= radius


init_default_sim("A")

if __name__ == "__main__":
    app.run(debug=True)
