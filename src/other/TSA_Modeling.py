import numpy as np
import pandas as pd


# Want to use data on a weekly basis to make forecasts/predictions
y = df['cost_per_watt'].resample('W').mean()

#resampling creates timeframes where there are no observations
y.dropna(inplace=True)

#take a look at the timeline
y.plot(figsize=(15, 6))
plt.show()


