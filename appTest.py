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

# open the json file and load into global ROUTES variable
with open("routes(1).json") as f:
   ROUTES = json.load(f)

buses = {}
for _, row in routes.iterrows():
    route_id = row['route_id']
    curr_data = busbreaks[busbreaks["routeName"] == route_id].copy()

    # quick filtering step to prevent data-type mismatching
    curr_data["stop_id"] = curr_data["stop_id"].astype(int)

    # now we want to iterate for each stop, creating another mini dataframe and computing the pdf on that.
    stop_seq = [int(s) for s in row["stop_sequence"].split(",")]

    bus = utils.busRoute(route_id, stop_seq)
    bus.setPath(ROUTES)
    bus.normalizePathAndStops()

    buses[route_id] = bus # adds the bus to our bus dictionary where the route_id is the index
    print('current route ', route_id )
    for stop_id in stop_seq:
        stop_data = curr_data[curr_data['stop_id'] == stop_id]
        stop_data = stop_data.reset_index(drop=True) # reset the indices so it looks nicer :)
        
        # so now at this step, we want to create the bus object with it's given pdf
        bus.generate_stop_pdf(stop_id, stop_data['break_duration']) # only send in the break_duration vector of the dataframe

app = Flask(__name__)

# --- Simulation state ---
CURRENT_ROUTE_ID = "LX"
NUM_BUSES = 4
SIM_SPEED = 1.0
LAST_UPDATE_TIME = time.time()
CURRENT_PDF_TYPE = "lognorm"
STOP_STATS = {}

# Per-frame data
EVENT_LOG = []
MAX_EVENTS = 100
def _new_metrics_state():
    return {
        "frame_count": 0,
        "bunch_score_sum": 0.0,
        "current": {
            "headway_score": 0.0,
            "cluster_score": 0.0,
            "bunch_score": 0.0,
            "bunch_score_avg": 0.0,
            "bunch_count": 0,
            "cac_count": 0,
            "liv_count": 0,
            "bunch_ratio": 0.0,
            "cluster_ratio": 0.0,
            "long_stop_count": 0,
        }
    }


def _reset_metrics_state():
    global METRICS
    METRICS = _new_metrics_state()


_reset_metrics_state()
SIM_TIMER = {
    "active": False,
    "duration": 0.0,
    "expires_at": None,
    "finished": False,
}


def _reset_timer_state():
    SIM_TIMER["active"] = False
    SIM_TIMER["duration"] = 0.0
    SIM_TIMER["expires_at"] = None
    SIM_TIMER["finished"] = False


def _timer_status(now=None):
    now = now or time.time()
    remaining = None
    if SIM_TIMER["active"] and SIM_TIMER["expires_at"]:
        remaining = max(0.0, SIM_TIMER["expires_at"] - now)
    elif SIM_TIMER["finished"]:
        remaining = 0.0
    return {
        "active": SIM_TIMER["active"],
        "finished": SIM_TIMER["finished"],
        "duration": SIM_TIMER["duration"],
        "remaining": remaining,
    }

# Active route data (updated on set_route)
PATH = []
NORM = []
STOPS = []
STOPS_BY_ID = {}
STOPS_SORTED = []  # stops sorted by normalized t along the path
runtime_buses = []


# --- Helpers ---
def normalize_route(route):
    route = np.array(route, dtype=float)
    if len(route) < 2:
        return route.tolist(), [0.0] * len(route)
    d = np.sqrt(np.sum(np.diff(route, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] == 0:
        s[-1] = 1.0
    s /= s[-1]
    return route.tolist(), s.tolist()


def map_stops_to_norm(path, norm, stops):
    """Map each stop to the closest point along the polyline path and compute its normalized t.

    Uses segment-wise projection instead of nearest vertex to avoid snapping to the wrong side
    of the path for stops near bends or intersections.
    """
    mapped = []
    if not path:
        # fallback: no path, return with t=0
        for i, s in enumerate(stops, start=1):
            if isinstance(s, dict):
                sx, sy = s.get("x"), s.get("y")
                sid = s.get("id", i)
                name = s.get("name", f"Stop {sid}")
            else:
                sx, sy = s[0], s[1]
                sid = i
                name = f"Stop {sid}"
            mapped.append({"id": int(sid), "x": float(sx), "y": float(sy), "t": 0.0, "name": name})
        return mapped

    for i, s in enumerate(stops, start=1):
        if isinstance(s, dict):
            sx, sy = s.get("x"), s.get("y")
            sid = s.get("id", i)
            name = s.get("name", f"Stop {sid}")
        else:
            sx, sy = s[0], s[1]
            sid = i
            name = f"Stop {sid}"

        best_dist2 = float("inf")
        best_t = 0.0
        # search each segment for the closest projection
        for k in range(len(path) - 1):
            x1, y1 = path[k]
            x2, y2 = path[k + 1]
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 1e-12:
                continue
            u = ((sx - x1) * dx + (sy - y1) * dy) / seg_len2
            u = max(0.0, min(1.0, u))
            cx = x1 + u * dx
            cy = y1 + u * dy
            dist2 = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)
            if dist2 < best_dist2:
                best_dist2 = dist2
                # interpolate t using normalized knot values
                if norm and k < len(norm) - 1:
                    best_t = float(norm[k] + u * (norm[k + 1] - norm[k]))
                else:
                    best_t = float(k / max(1, len(path) - 1))

        mapped.append({
            "id": int(sid),
            "x": float(sx),
            "y": float(sy),
            "t": best_t,
            "name": name,
        })

    return mapped


# Utility: find the next stop ahead of a given normalized position
def next_stop_from_pos(pos):
    global STOPS_SORTED
    if not STOPS_SORTED:
        return None, None
    eps = 1e-6
    for st in STOPS_SORTED:
        if st["t"] > pos + eps:
            return int(st["id"]), float(st["t"])
    st0 = STOPS_SORTED[0]
    return int(st0["id"]), float(st0["t"]) if isinstance(st0["t"], (int, float)) else 0.0


def _build_runtime_buses(count):
    global runtime_buses
    route_id = CURRENT_ROUTE_ID
    model = buses.get(route_id, next(iter(buses.values())))
    # ensure pdf type matches current selection
    try:
        model.set_pdf_type(CURRENT_PDF_TYPE)
    except Exception:
        pass

    new_list = []
    for i in range(max(1, int(count))):
        bus_pos = i / max(1, int(count))
        tid, tt = next_stop_from_pos(bus_pos)
        new_list.append({
            "id": f"Bus {i+1}",
            "route_id": route_id,
            "pos": bus_pos,
            "speed": 0.002 + random.random() * 0.002,
            "wait": 0.0,
            "at_stop": False,
            "dwell_time": 0.0,
            "last_pos": None,
            "last_stop_id": None,
            "model": model,
            "target_id": tid,
            "target_t": tt,
        })
    runtime_buses = new_list


def interpolate(points, norm, t):
    if not points:
        return (0.0, 0.0)
    if len(points) == 1 or not norm:
        return tuple(points[0])
    target = float(t % 1.0)
    for i in range(len(norm) - 1):
        a, b = norm[i], norm[i + 1]
        if a <= target <= b and b > a:
            frac = (target - a) / (b - a)
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            return (x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac)
    return tuple(points[-1])


def check_segment_near_stop(x1, y1, x2, y2, sx, sy, radius=6.0):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((x1 - sx) ** 2 + (y1 - sy) ** 2) ** 0.5 <= radius
    t = ((sx - x1) * dx + (sy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nx = x1 + t * dx
    ny = y1 + t * dy
    return ((nx - sx) ** 2 + (ny - sy) ** 2) ** 0.5 <= radius


def _init_route_state(route_id):
    global CURRENT_ROUTE_ID, PATH, NORM, STOPS, STOPS_BY_ID, STOPS_SORTED, STOP_STATS, runtime_buses
    if route_id not in ROUTES:
        route_id = list(ROUTES.keys())[0]
    CURRENT_ROUTE_ID = route_id

    data = ROUTES[route_id]
    path, norm = normalize_route(data["path"])
    stops = map_stops_to_norm(path, norm, data["stops"])

    PATH, NORM, STOPS = path, norm, stops
    STOPS_BY_ID = {int(s["id"]): s for s in STOPS}
    STOPS_SORTED = sorted(STOPS, key=lambda s: s["t"]) if STOPS else []
    STOP_STATS = {int(s["id"]): {"count": 0, "sum": 0.0} for s in STOPS}

    # build buses using current NUM_BUSES
    _build_runtime_buses(NUM_BUSES)
    _reset_metrics_state()
    _reset_timer_state()
    global LAST_UPDATE_TIME
    LAST_UPDATE_TIME = time.time()

# unnecessary, we already normalzized
# Normalize every route once to speed up switching
for key, data in ROUTES.items():
    p, n = normalize_route(data["path"])
    s = map_stops_to_norm(p, n, data["stops"])
    data["path"], data["norm"], data["stops"] = p, n, s

# Default to LX if available
_init_route_state("LX" if "LX" in ROUTES else list(ROUTES.keys())[0])


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set_route", methods=["POST"])
def set_route():
    data = request.get_json(silent=True) or {}
    route_id = data.get("route_id")
    if not route_id or route_id not in ROUTES:
        return jsonify({"error": "Invalid route_id"}), 400
    _init_route_state(route_id)
    return jsonify({"ok": True, "route": route_id})


@app.route("/positions")
def positions():
    global LAST_UPDATE_TIME
    now = time.time()
    dt = (now - LAST_UPDATE_TIME) * SIM_SPEED
    LAST_UPDATE_TIME = now
    timer_paused = False
    if SIM_TIMER["active"] and SIM_TIMER["expires_at"]:
        remaining_after = SIM_TIMER["expires_at"] - now
        if remaining_after <= 0:
            allowed = max(0.0, dt + remaining_after)
            dt = allowed
            SIM_TIMER["active"] = False
            SIM_TIMER["expires_at"] = None
            SIM_TIMER["finished"] = True
            timer_paused = dt == 0.0
    elif SIM_TIMER["finished"]:
        dt = 0.0
        timer_paused = True

    active = [b for b in runtime_buses if b["route_id"] == CURRENT_ROUTE_ID]
    path = PATH
    stops = STOPS

    def crossed(prev, new, target):
        prev = prev % 1.0
        new = new % 1.0
        target = target % 1.0
        if prev <= new:
            return prev <= target <= new
        return target >= prev or target <= new

    for bus in active:
        if bus["wait"] > 0:
            bus["wait"] = max(0.0, bus["wait"] - dt)
            if bus["wait"] == 0.0:
                bus["at_stop"] = False
                # choose next nearest stop ahead after finishing dwell
                if STOPS_SORTED:
                    # pos is at the stop's t; move target to the next ahead
                    eps = 1e-6
                    current_pos = bus["pos"]
                    # find first stop with t > current_pos + eps, else wrap
                    next_stop = None
                    for st in STOPS_SORTED:
                        if st["t"] > current_pos + eps:
                            next_stop = st
                            break
                    if next_stop is None:
                        next_stop = STOPS_SORTED[0]
                    bus["target_id"] = int(next_stop["id"]) if next_stop else None
                    bus["target_t"] = float(next_stop["t"]) if next_stop else None
            continue

        if bus["last_pos"] is None:
            bus["last_pos"] = interpolate(path, NORM, bus["pos"])
        x1, y1 = bus["last_pos"]

        pos_prev = bus["pos"]
        pos_new = (pos_prev + bus["speed"] * dt) % 1.0

        sid = bus.get("target_id")
        target_t = bus.get("target_t")
        st = STOPS_BY_ID.get(int(sid)) if sid is not None else None
        if st is None:
            # pick a target if missing
            if STOPS_SORTED:
                st0 = STOPS_SORTED[0]
                bus["target_id"] = int(st0["id"])
                bus["target_t"] = float(st0["t"]) if isinstance(st0["t"], (int, float)) else 0.0
                st = st0
                target_t = bus["target_t"]

        if st is not None and target_t is not None and crossed(pos_prev, pos_new, float(target_t)):
            # snap to stop and dwell
            bus["pos"] = float(target_t)
            sx, sy = st["x"], st["y"]
            bus["last_pos"] = (sx, sy)
            if not (bus["at_stop"] and bus.get("last_stop_id") == int(st["id"])):
                try:
                    dwell = float(bus["model"].sampleStop(int(st["id"])) )
                except Exception:
                    dwell = 15.0
                bus["wait"] = dwell
                bus["dwell_time"] = dwell
                bus["at_stop"] = True
                bus["last_stop_id"] = int(st["id"]) 
                # update running stop stats
                try:
                    sid_int = int(st["id"]) 
                    if sid_int in STOP_STATS:
                        STOP_STATS[sid_int]["count"] += 1
                        STOP_STATS[sid_int]["sum"] += dwell
                except Exception:
                    pass
                # long stop metric (>= 120s)
                try:
                    if float(dwell) >= 120.0:
                        METRICS["current"]["long_stop_count"] = METRICS["current"].get("long_stop_count", 0) + 1
                except Exception:
                    pass
                EVENT_LOG.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "bus_id": bus["id"],
                    "stop_id": int(st["id"]),
                    "stop_name": st.get("name", f"Stop {int(st['id'])}"),
                    "dwell": dwell,
                })
                if len(EVENT_LOG) > MAX_EVENTS:
                    del EVENT_LOG[: len(EVENT_LOG) - MAX_EVENTS]
            # do not choose next target here; will choose after dwell completes
            continue

        # normal movement (no stop crossed)
        bus["pos"] = pos_new
        x2, y2 = interpolate(path, NORM, bus["pos"])
        bus["last_pos"] = (x2, y2)

    # Build lightweight response
    buses_out = []
    for b in active:
        status = None
        if b["wait"] > 0 and b["dwell_time"] > 0:
            status = {
                "stop_id": b.get("last_stop_id"),
                "remaining": round(b["wait"], 2),
                "progress": max(0.0, min(1.0, 1.0 - b["wait"] / b["dwell_time"]))
            }
            # include label if available
            for st in STOPS:
                if int(st["id"]) == status["stop_id"]:
                    status["stop_name"] = st.get("name", f"Stop {status['stop_id']}")
                    break
        buses_out.append({"id": b["id"], "pos": b["pos"], "status": status})

    # Update metrics based on current positions (only if sim is advancing)
    if not timer_paused:
        try:
            n = len(active)
            if n >= 2:
                positions = sorted([b["pos"] % 1.0 for b in active])
                gaps = []
                for i in range(n - 1):
                    gaps.append(positions[i + 1] - positions[i])
                gaps.append((positions[0] + 1.0) - positions[-1])
                ideal = 1.0 / n
                bunch_thresh = 0.5 * ideal
                cluster_thresh = 0.25 * ideal
                bunch_count = sum(1 for g in gaps if g < bunch_thresh)
                bunch_ratio = bunch_count / n
                cluster_count = sum(1 for g in gaps if g < cluster_thresh)
                cluster_ratio = cluster_count / n

                closeness = [max(0.0, 1.0 - (g / ideal)) for g in gaps]
                headway_score = sum(closeness) / n

                METRICS["current"].update({
                    "headway_score": float(headway_score),
                    "bunch_score": float(bunch_ratio),
                    "cluster_score": float(cluster_ratio),
                    "bunch_count": int(bunch_count),
                    "bunch_ratio": float(bunch_ratio),
                    "cluster_ratio": float(cluster_ratio),
                })
            else:
                METRICS["current"].update({
                    "headway_score": 0.0,
                    "bunch_score": 0.0,
                    "cluster_score": 0.0,
                    "bunch_count": 0,
                    "bunch_ratio": 0.0,
                    "cluster_ratio": 0.0,
                })
        except Exception:
            pass

        METRICS["frame_count"] += 1
        curr_bunch = METRICS["current"].get("bunch_score", 0.0)
        METRICS["bunch_score_sum"] += curr_bunch
        if METRICS["frame_count"] > 0:
            METRICS["current"]["bunch_score_avg"] = (
                METRICS["bunch_score_sum"] / METRICS["frame_count"]
            )
        else:
            METRICS["current"]["bunch_score_avg"] = 0.0

    timer_info = _timer_status(now)

    return jsonify({
        "path": PATH,
        "norm": NORM,
        "stops": STOPS,
        "buses": buses_out,
        "route": CURRENT_ROUTE_ID,
        "timer": timer_info,
    })


@app.route("/meta")
def meta():
    return jsonify({
        "route_id": CURRENT_ROUTE_ID,
        "num_buses": len([b for b in runtime_buses if b["route_id"] == CURRENT_ROUTE_ID]),
        "num_stops": len(STOPS),
        "speed_factor": SIM_SPEED,
        "timer": _timer_status(),
    })


@app.route("/events")
def events():
    return jsonify(EVENT_LOG[-20:])


@app.route("/metrics")
def metrics():
    return jsonify(METRICS["current"]) 


@app.route("/speed", methods=["POST"])
def set_speed():
    global SIM_SPEED
    data = request.get_json(silent=True) or {}
    try:
        factor = float(data.get("factor", 1.0))
    except Exception:
        return jsonify({"error": "Invalid factor"}), 400
    SIM_SPEED = max(0.1, min(factor, 200.0))
    return jsonify({"ok": True, "speed_factor": SIM_SPEED})

@app.route("/pdf", methods=["POST"])
def set_pdf_type():
    global CURRENT_PDF_TYPE
    data = request.get_json(silent=True) or {}
    pdf_type = str(data.get("type", "")).strip().lower()
    allowed = {"lognorm", "exponential", "uniform", "constant", "uniform_nosc", "constant_nosc"}
    if pdf_type not in allowed:
        return jsonify({"error": f"Invalid pdf type. Allowed: {sorted(list(allowed))}"}), 400
    CURRENT_PDF_TYPE = pdf_type
    # update all bus models to the chosen type
    for bus in buses.values():
        try:
            bus.set_pdf_type(pdf_type)
        except Exception:
            # continue even if a specific route lacks data for a type
            continue
    _build_runtime_buses(NUM_BUSES)
    _reset_metrics_state()
    _reset_timer_state()
    global LAST_UPDATE_TIME
    LAST_UPDATE_TIME = time.time()
    return jsonify({"ok": True, "pdf_type": CURRENT_PDF_TYPE})


@app.route("/timer", methods=["POST"])
def configure_timer():
    data = request.get_json(silent=True) or {}
    if "minutes" not in data:
        return jsonify({"error": "Missing 'minutes' field"}), 400
    try:
        minutes = float(data.get("minutes", 0))
    except Exception:
        return jsonify({"error": "Invalid minutes value"}), 400

    if minutes <= 0:
        _reset_timer_state()
        return jsonify({"ok": True, "timer": _timer_status()})

    duration = max(0.0, minutes) * 60.0
    SIM_TIMER["active"] = True
    SIM_TIMER["duration"] = duration
    SIM_TIMER["expires_at"] = time.time() + duration
    SIM_TIMER["finished"] = False
    return jsonify({"ok": True, "timer": _timer_status()})

@app.route("/buses", methods=["POST"])
def set_bus_count():
    global NUM_BUSES
    data = request.get_json(silent=True) or {}
    try:
        count = int(data.get("count", NUM_BUSES))
    except Exception:
        return jsonify({"error": "Invalid count"}), 400
    count = max(1, min(30, count))
    NUM_BUSES = count
    _build_runtime_buses(NUM_BUSES)
    return jsonify({"ok": True, "num_buses": NUM_BUSES})

@app.route("/stop_stats")
def stop_stats():
    model = buses.get(CURRENT_ROUTE_ID, None)
    out = []
    stops_iter = STOPS_SORTED if STOPS_SORTED else STOPS
    for st in stops_iter:
        sid = int(st["id"])
        name = st.get("name", f"Stop {sid}")
        theory = None
        try:
            if model is not None and sid in getattr(model, 'stops', {}):
                theory = float(model.stops[sid].get('mean'))
        except Exception:
            theory = None
        sim_mean = None
        stats = STOP_STATS.get(sid, {"count": 0, "sum": 0.0})
        if stats["count"] > 0:
            sim_mean = stats["sum"] / stats["count"]
        diff = None
        if theory is not None and sim_mean is not None:
            diff = sim_mean - theory
        out.append({
            "id": sid,
            "name": name,
            "mean_theory": theory,
            "mean_sim": sim_mean,
            "diff": diff,
        })
    return jsonify(out)

if __name__ == "__main__":
    # launch development server
    app.run(debug=True)
