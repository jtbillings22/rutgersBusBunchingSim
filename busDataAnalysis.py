import numpy as np
import pandas as pd
import os 
from scipy import stats
import matplotlib.pyplot as plt


busbreaks = pd.read_csv(os.path.join('rutgers_bus_data','bus_breaks.csv'))
routes = pd.read_csv(os.path.join('rutgers_bus_data','routes.csv'))
vehicles = pd.read_csv(os.path.join('rutgers_bus_data','vehicles.csv'),dtype={"routeName": str})
vehicles['routeName'] = vehicles['routeName'].str.replace(' Route', '', regex=False)


# goal is to merge bus breaks and vehicles data to assign routeId's to the bus break events.

busbreaks = busbreaks.merge(vehicles[['id', 'routeName']], on='id', how='left')

#print(busbreaks.columns.to_list())
#print(busbreaks.head)   

#print(busbreaks['routeName'])

# now we step into probalistic modeling;

subset = busbreaks[
    (busbreaks['routeName'] == 'LX') &
    (busbreaks['stop_id'] == 1) # this is the CA student center
]


def displayPDF(pdf,x):
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    x = np.linspace(data.min(), data.max(), 300)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)


    plt.plot(x, pdf, 'r-', lw=2, label='LX Fitted Lognormal PDF')
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.6)
    plt.xticks(np.arange(0, data.max() + 100, 100))      
    plt.xlabel('Break Duration (seconds)')
    plt.ylabel('Probability Density')
    plt.title('Fitted Lognormal PDF')

    # shade under the curve up until the mean, and add a vertical line at mean
    plt.fill_between(x, 0, pdf, where=(x <= mean), color='skyblue', alpha=0.5, label='Area below mean')
    plt.axvline(mean, color='k', linestyle='--', label=f'Mean = {mean:.1f}s')
    plt.savefig("results/pdf_LX_with_mean.png", dpi=300, bbox_inches='tight')
    plt.show()

# Example: fit a normal distribution
data = subset['break_duration'].dropna()

shape, loc, scale = stats.lognorm.fit(data, floc=0)
x = np.linspace(data.min(), data.max(), 200)
pdf = stats.lognorm.pdf(x, shape, loc, scale)
displayPDF(pdf,x)

mean = stats.lognorm.mean(shape, loc=loc, scale=scale)
median = stats.lognorm.median(shape, loc=loc, scale=scale)
#print(f"Mean: {mean:.2f}, Median: {median:.2f}")
#samples = stats.lognorm.rvs(shape, loc, scale, size=10)
#print(samples)

pdf = pdf / pdf.sum()
displayPDF(pdf,x)
samples = np.random.choice(x, size=10, p=pdf)
#print("New samples mean", np.mean(samples), "new samples median", np.median(samples))
#print(samples)



# okay so now we have a way of generating the PDF for a given route, now we want to go ahead and generate an array of PDF's of
# busbreak duration for each bus at each stop. Then we can also generalize these pdfs into a single one for each stop, however
# i feel that would be inaccurate as higher frequency stops outweigh lower frequency one, so easier to create a pdf for
# each stop, and draw the sample from there when we reach it. 
routes['route_id'] = routes['route_id'].astype(str).str.strip().str.upper()
vehicles['routeName'] = vehicles['routeName'].astype(str).str.strip().str.upper()
routes = routes[routes["route_id"].isin(vehicles['routeName'])]

# now we have a nice clean routes df, so for each stop, we can generate a list corresponding to the stop

# practice
stop = {}
stop_id = 1
stop[stop_id] = pdf

samples = np.random.choice(x, size = 10, p=stop[1])
print(samples)

        
