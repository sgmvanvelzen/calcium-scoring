# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
from collections import deque
from visdom import Visdom


class LearningCurve:
    """Plots multiple lines with Visdom by sending one data point per line each time"""
    def __init__(self, title, legend=('Training', 'Validation'), moving_average=1, server=None, env=None, **kwargs):
        assert moving_average >= 1

        if server is None:
            self.visdom = Visdom(use_incoming_socket=False)
        else:
            if server.count(':') > 1:
                server, port = server.rsplit(':', 1)
                self.visdom = Visdom(server=server, port=port, use_incoming_socket=False)
            else:
                try:
                    self.visdom = Visdom(port=int(server), use_incoming_socket=False)
                except ValueError:
                    self.visdom = Visdom(server=server, use_incoming_socket=False)

        self.window = None

        legend = tuple(legend)
        self.num_lines = len(legend)
        self.submitted = 0

        self.moving_average = int(moving_average)
        self.previous_values = deque()

        self.env = 'UOK_CalciumScoring_{}'.format(env)
        self.options = kwargs
        self.options['title'] = title
        self.options['legend'] = list(legend)

    def post(self, values, step=None):
        """Adds a new data point to each line in the graph"""
        y = np.asarray(values).flatten()
        if y.size != self.num_lines:
            raise ValueError('Expected {} values, but got {} values'.format(self.num_lines, y.size))

        # Compute moving average if that is enabled
        if self.moving_average > 1:
            self.previous_values.append(y)
            if len(self.previous_values) < self.moving_average:
                return
            elif len(self.previous_values) > self.moving_average:
                self.previous_values.popleft()

            y = np.mean(self.previous_values, axis=0)

        # Submit values to visdom server
        if step is None:
            step = self.submitted + self.moving_average

        if self.num_lines == 1:
            x = np.asarray(step).reshape(1)
        else:
            x = np.repeat(step, self.num_lines).reshape(1, -1)
            y = y.reshape(1, -1)

        if self.window:
            self.visdom.line(X=x, Y=y, win=self.window, env=self.env, opts=self.options, update='append')
        else:
            self.window = self.visdom.line(X=x, Y=y, env=self.env, opts=self.options)

        self.submitted += 1
