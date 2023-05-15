"""A set of common values and functions that are globaly available."""
import sys
import functools

# Common Modules
from ottermatics.configuration import *
from ottermatics.logging import LoggingMixin
from urllib.request import urlopen

import numpy


class OtterLog(LoggingMixin):
    pass


log = OtterLog()
log.info("Starting Ottermatics Enviornment")


class inst_vectorize(numpy.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def is_ec2_instance():
    """Check if an instance is running on AWS."""
    result = False
    meta = "http://169.254.169.254/latest/meta-data/public-ipv4"
    try:
        result = urlopen(meta, timeout=5.0).status == 200
        return True
    except:
        return False
    return False


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        size += sum([get_size(i, seen) for i in obj])
    return size


# Constants
g = gravity = 9.80665  # m/s2
G_grav_constant = 6.67430e-11  # m3/kgs
speed_of_light = 299792458  # m/s
u_planck = 6.62607015e-34  # Js
R_univ_gas = 8.314462618  # J/molkg
mass_electron = 9.1093837015e-31  # kg
mass_proton = 1.67262192369e-27
mass_neutron = 1.67492749804e-27
r_electron = 2.8179403262e-15
Kboltzman = 1.380649e-23  # J⋅K−1
