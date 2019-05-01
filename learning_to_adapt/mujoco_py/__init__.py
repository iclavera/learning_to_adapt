'''
Code from rllab
'''

from .mjviewer import MjViewer
from .mjcore import MjModel
from .mjcore import register_license
import os
from .mjconstants import *

register_license(os.path.join(os.path.expanduser('~'), '.mujoco/mjkey.txt'))
