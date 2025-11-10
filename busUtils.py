import numpy as np
import pandas as pd
import os 
from scipy import stats
import matplotlib.pyplot as plt
import json

class busRoute:
    def __init__(self, route_id, stop_seq):
        self.route_id = route_id
        self.stops = {}
        self.norm = [] # normalized path distance
        self.path = {} # original path 
        self.pathStops = {} # coordinates of all the stops
        self.pdfType = "lognorm"
        self.stop_sequence = stop_seq

    # initilize bus stops and paths for it's route. Accesses the routes dict which is organized as:
    # { "route_id": 
    #   {
    #       "path": [list of path coordinates], 
    #       "stops: {dictionary of stops} "
    #    }
    #   . . .
    # }
    def set_pdf_type(self, pdf_type):
        self.pdfType = pdf_type
        self.updatePdf(pdf_type)

    def updatePdf(self, pdf_type):
        self.pdfType = pdf_type

        sc_ids = [1,10,13,17]

        def fit_lognorm(data_series):
            d = pd.to_numeric(data_series, errors="coerce").dropna()
            if len(d) < 3 or d.min() >= d.max():
                return None
            shape, loc, scale = stats.lognorm.fit(d, floc=0)
            x = np.linspace(max(1e-6, d.min()), d.max(), 200)
            pdf = stats.lognorm.pdf(x, shape, loc, scale)
            pdf = pdf / pdf.sum()
            mean = stats.lognorm.mean(shape, loc, scale)
            median = stats.lognorm.median(shape, loc, scale)
            return {
                'shape': shape,
                'loc': loc,
                'scale': scale,
                'x': x,
                'pdf': pdf,
                'mean': mean,
                'median': median
            }

        def fit_expon(data_series):
            d = pd.to_numeric(data_series, errors="coerce").dropna()
            if len(d) < 3 or d.min() >= d.max():
                return None
            loc, scale = stats.expon.fit(d, floc=0)
            x = np.linspace(max(0.0, d.min()), d.max(), 200)
            pdf = stats.expon.pdf(x, loc, scale)
            pdf = pdf / pdf.sum()
            mean = loc + scale
            median = stats.expon.median(loc=loc, scale=scale)
            return {
                'shape': None,
                'loc': loc,
                'scale': scale,
                'x': x,
                'pdf': pdf,
                'mean': mean,
                'median': median
            }

        # Iterate through each stop and update its PDF according to the selected type
        for stop_id, stop in list(self.stops.items()):
            data = stop.get('data', pd.Series([], dtype=float))
            is_sc = int(stop_id) in sc_ids

            # Determine effective type (handle *_nosc variants)
            eff_type = pdf_type
            if pdf_type == 'uniform_nosc':
                eff_type = 'lognorm' if is_sc else 'uniform'
            elif pdf_type == 'constant_nosc':
                eff_type = 'lognorm' if is_sc else 'constant'

            if eff_type == 'lognorm':
                res = fit_lognorm(data)
                if res is None:
                    # Fallback to a light uniform if insufficient/degenerate data
                    x = np.linspace(15, 60, 200)
                    pdf = np.ones_like(x, dtype=float)
                    pdf /= pdf.sum()
                    self.stops[stop_id].update({
                        'shape': None,
                        'loc': None,
                        'scale': None,
                        'x': x,
                        'pdf': pdf,
                        'mean': float(np.sum(x * pdf)),
                        'median': float(np.median(x))
                    })
                else:
                    self.stops[stop_id].update(res)

            elif eff_type == 'exponential':
                res = fit_expon(data)
                if res is None:
                    # Fallback to uniform if insufficient data
                    x = np.linspace(15, 60, 200)
                    pdf = np.ones_like(x, dtype=float)
                    pdf /= pdf.sum()
                    self.stops[stop_id].update({
                        'shape': None,
                        'loc': None,
                        'scale': None,
                        'x': x,
                        'pdf': pdf,
                        'mean': float(np.sum(x * pdf)),
                        'median': float(np.median(x))
                    })
                else:
                    self.stops[stop_id].update(res)

            elif eff_type == 'uniform':
                x = np.linspace(15, 60, 200)
                pdf = np.ones_like(x, dtype=float)
                pdf /= pdf.sum()
                self.stops[stop_id].update({
                    'shape': None,
                    'loc': None,
                    'scale': None,
                    'x': x,
                    'pdf': pdf,
                    'mean': float(np.sum(x * pdf)),
                    'median': float(np.median(x))
                })

            elif eff_type == 'constant':
                x = np.array([20.0])
                pdf = np.array([1.0])
                self.stops[stop_id].update({
                    'shape': None,
                    'loc': None,
                    'scale': None,
                    'x': x,
                    'pdf': pdf,
                    'mean': 20.0,
                    'median': 20.0
                })

            else:
                # Unknown type: keep existing and continue
                continue

    def setPath(self, routes):
        self.path = routes[self.route_id]["path"]
        self.pathStops = routes[self.route_id]["stops"]

    def normalizePathAndStops(self):
        path = np.array(self.path)

        d = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        s = np.concatenate([[0], np.cumsum(d)])
        s /= s[-1]  # normalize to [0,1]
        # intialize path to the new normalized path
        self.norm = s.tolist()
        #print(" Successfully normalized the {self.route_id} path")

        
    def sampleStop(self, stop_id):
        stop = self.stops[stop_id]
        pdf = stop['pdf']
        x = stop.get('x')
        if x is None:
            data = stop.get('data')
            # Fallback to data-derived support if x missing
            if data is not None and len(data) > 0:
                x = np.linspace(float(np.min(data)), float(np.max(data)), len(pdf))
            else:
                x = np.linspace(15, 60, len(pdf))
        sample = np.random.choice(x, size=1, p=pdf)
        return float(sample[0])
    
    def generate_stop_pdf(self, stop_id, data): # requires stop_sequence and the data
        # computes the dataset into numeric values
        data = pd.to_numeric(data, errors ="coerce").dropna() 
        
        # guard 1 — skip if no numeric data
        if len(data) == 0:
            print(f"[SKIP] No valid numeric data for stop {stop_id} on route {self.route_id}")
            return
        
        # guard 2 — skip if all values identical (lognorm.fit fails on constant arrays)
        if data.min() == data.max():
            print(f"[SKIP] Constant data for stop {stop_id} on route {self.route_id}")
            return

        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        x = np.linspace(data.min(), data.max(), 200)
        pdf = stats.lognorm.pdf(x, shape, loc, scale)
        pdf = pdf / pdf.sum()
        mean = stats.lognorm.mean(shape, loc, scale)
        median = stats.lognorm.median(shape,loc,scale)

        # assigns values into stops dictionary - each stop_id has its associated PDF, as well as it's components
        self.stops[stop_id] = {
            'shape': shape,
            'loc': loc,
            'scale': scale,
            'x': x,
            'pdf': pdf,
            'route_name': self.route_id,
            'data': data,
            "mean": mean,
            "median": median
        }

    def print_path(self):
        print(self.path)

    def display_stop_pdf(self, stop_id):
        if stop_id not in self.stops:
            print(f"[SKIP PLOT] No fitted PDF for stop {stop_id} on route {self.route_id}")
            return

        stop = self.stops[stop_id]
        route_name = stop['route_name']
        x = stop.get('x')
        pdf = stop.get('pdf')
        data = stop['data']

        # plot empirical data histogram (empirical distribution)
        plt.hist(data, bins=30, density=True, alpha=0.5, label='Empirical Data', color='steelblue', edgecolor='black')
        
        # Determine effective type for labeling (supports *_nosc variants)
        sc_ids = [1,10,13,17]
        is_sc = int(stop_id) in sc_ids
        eff_type = self.pdfType
        if self.pdfType == 'uniform_nosc':
            eff_type = 'lognorm' if is_sc else 'uniform'
        elif self.pdfType == 'constant_nosc':
            eff_type = 'lognorm' if is_sc else 'constant'

        # Overlay current fitted/assigned PDF depending on type
        label_map = {
            'lognorm': 'Lognormal fit',
            'uniform': 'Uniform (15–60s)',
            'constant': 'Constant (20s)',
            'exponential': 'Exponential fit'
        }
        label = label_map.get(eff_type, 'Assigned PDF')

        if x is not None and pdf is not None and len(x) == len(pdf) and len(x) > 1 and eff_type != 'constant':
            plt.plot(x, pdf, 'r-', lw=2, label=f'{stop_id} {label}')
        elif eff_type == 'constant' and x is not None and len(np.atleast_1d(x)) >= 1:
            x0 = float(np.atleast_1d(x)[0])
            plt.axvline(x0, color='red', linestyle='-', lw=2, label=f'{stop_id} {label}')
        else:
            # Fallback draw using zero line if support missing
            xs = np.linspace(float(np.min(data)) if len(data) else 15,
                             float(np.max(data)) if len(data) else 60, 200)
            plt.plot(xs, np.zeros_like(xs), 'r-', lw=2, label=f'{stop_id} {label}')
        plt.legend()
        plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
        plt.xticks(np.arange(0, data.max() + 100, 100))      
        plt.xlabel('Break Duration (seconds)')
        plt.ylabel('Probability Density')
        plt.title(f'{route_name} - {stop_id} (Empirical vs {label})')
        # Guard mean computation when x/pdf missing
        if x is None or pdf is None or len(np.atleast_1d(pdf)) == 0:
            mean = float(np.mean(data)) if len(data) else 0.0
        else:
            x_arr = np.atleast_1d(x)
            pdf_arr = np.atleast_1d(pdf)
            # align sizes if needed
            if len(x_arr) != len(pdf_arr) and len(x_arr) > 1:
                n = min(len(x_arr), len(pdf_arr))
                x_arr = x_arr[:n]
                pdf_arr = pdf_arr[:n]
            mean = float(np.sum(x_arr * pdf_arr) / np.sum(pdf_arr))
        plt.axvline(mean, color='k', linestyle='--', label=f'Mean = {mean:.1f}s')

        plt.show()

    def generate_pdf_image(self, stop_id, output_dir='results'):
        if stop_id not in self.stops:
            print(f"[SKIP PLOT] No fitted data for stop {stop_id} on route {self.route_id}")
            return

        stop = self.stops[stop_id]
        route_name = stop['route_name']
        data = stop['data']
        data = data[data > 1e-3]  # remove zeros / near-zeros
        if len(data) < 3:
            print(f"[SKIP] Too few data points to plot for stop {stop_id} on route {route_name}")
            return

        # Range for the fitted curves
        x = np.linspace(data.min(), data.max(), 400)

        # fit log-normal pdf
        shape_l, loc_l, scale_l = stats.lognorm.fit(data, floc=0)
        pdf_l = stats.lognorm.pdf(x, shape_l, loc_l, scale_l)

        # fit gamma pdf
        shape_g, loc_g, scale_g = stats.gamma.fit(data, floc=0)
        pdf_g = stats.gamma.pdf(x, shape_g, loc_g, scale_g)

        # fit exponential pdf

        loc_2, scale_2 = stats.expon.fit(data,floc=0)
        pdf_2 = stats.expon.pdf(x, loc_2, scale_2)


        # plot the empircal data and both fits
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=30, density=True, alpha=0.4,
                color='gray', edgecolor='black', label='Empirical Data')

        plt.plot(x, pdf_l, 'r-', lw=2, alpha=0.8, label='Lognormal fit')
        plt.plot(x, pdf_g, 'b--', lw=2, alpha=0.8, label='Gamma fit')
        plt.plot(x, pdf_2, 'g--', lw=2, alpha=0.8, label='Exponential fit')


        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xscale('log')
        plt.xlabel('Break Duration (seconds)')
        plt.ylabel('Probability Density')
        plt.title(f'{route_name} – Stop {stop_id} (Lognormal vs Gamma vs Exponential)')

        # saving
        os.makedirs(output_dir, exist_ok=True)
        route_dir = os.path.join(output_dir, f'route_{route_name}')
        os.makedirs(route_dir, exist_ok=True)
        save_path = os.path.join(route_dir, f"compare_{route_name}_{stop_id}.png")

        if os.path.exists(save_path):
            print(f"[SKIP SAVE] File already exists → {save_path}")
            plt.close()
            return

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[SAVED] Comparison plot (Lognormal vs Gamma) → {save_path}")



