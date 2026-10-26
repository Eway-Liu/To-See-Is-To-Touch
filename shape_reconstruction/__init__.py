"""Public sensor API; load hardware and visualization dependencies on demand."""
from importlib import import_module


__all__ = ['Camera', 'Sensor', 'Visualizer']


def __getattr__(name):
    modules = {
        'Camera': '.core.camera',
        'Sensor': '.core.sensor',
        'Visualizer': '.visualization.visualizer',
    }
    if name not in modules:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
