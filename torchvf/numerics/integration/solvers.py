# Copyright 2022 Ryan Peters
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################
# For the solvers I am referencing:
# - https://mathworld.wolfram.com/Runge-KuttaMethod.html
# - https://numerary.readthedocs.io/en/latest/dormand-prince-method.html
# - https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
# 
########################################################

import torch


class Euler:
    """ Fixed Euler method. """
    def __init__(self, f):
        self.f = f 

    def step(self, points, dx):
        k1 = self.f(points)

        out = points + dx * k1

        return out, None


class Midpoint:
    """ Fixed Midpoint method. """
    def __init__(self, f):
        self.f = f

    def step(self, points, dx):
        k1 = self.f(points)
        k2 = self.f(points + dx * k1 / 2)

        out = points + dx * k2

        return out, None


class RungeKutta:
    """ Fixed 4th order Runge Kutta method. """
    def __init__(self, f):
        self.f = f 

    def step(self, points, dx):
        k1 = self.f(points)
        k2 = self.f(points + dx * k1 / 2)
        k3 = self.f(points + dx * k2 / 2)
        k4 = self.f(points + dx * k3)

        out = points + (dx / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

        return out, None 





