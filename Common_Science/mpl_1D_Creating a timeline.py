"""
https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html#sphx-glr-gallery-lines-bars-and-markers-timeline-py

"""
# from datetime import datetime
#
# # In case the above fails, e.g. because of missing internet connection
# # use the following lists as fallback.
# releases = ['2.2.4', '3.0.3', '3.0.2', '3.0.1', '3.0.0', '2.2.3',
#             '2.2.2', '2.2.1', '2.2.0', '2.1.2', '2.1.1', '2.1.0',
#             '2.0.2', '2.0.1', '2.0.0', '1.5.3', '1.5.2', '1.5.1',
#             '1.5.0', '1.4.3', '1.4.2', '1.4.1', '1.4.0']
# dates = ['2019-02-26', '2019-02-26', '2018-11-10', '2018-11-10',
#          '2018-09-18', '2018-08-10', '2018-03-17', '2018-03-16',
#          '2018-03-06', '2018-01-18', '2017-12-10', '2017-10-07',
#          '2017-05-10', '2017-05-02', '2017-01-17', '2016-09-09',
#          '2016-07-03', '2016-01-10', '2015-10-29', '2015-02-16',
#          '2014-10-26', '2014-10-18', '2014-08-26']
#
# dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]  # Convert strs to dates.
# dates, releases = zip(*sorted(zip(dates, releases)))  # Sort by increasing date.
#

import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 2 * np.pi, 1024)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

fig, ax = plt.subplots()
im = ax.imshow(data2d)
ax.set_title('Pan on the colorbar to shift the color mapping\n'
             'Zoom on the colorbar to scale the color mapping')

fig.colorbar(im, ax=ax, label='Interactive colorbar')

plt.show()