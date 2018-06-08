name = 'pycnnum'

try:
    from pycnnum import cn2num, num2cn
except ImportError:
    from .pycnnum import cn2num, num2cn
