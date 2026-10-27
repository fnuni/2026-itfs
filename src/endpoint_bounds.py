"""Credibility and OWA bounds identified by finitely many elicited endpoints.

Each supplied (ceiling, sum of planning endpoints) is an exact feasibility
oracle for the chosen plan. Sentinels 0 and 1 bound all credibilities. No
membership completion is inferred from the supplied answers.
"""
from exact_fff import cell_owa

def risk_interval(level_durations, limit, empty=False):
    if empty:
        return (0., 0.)
    levels = sorted(level_durations)
    if any(not 0 <= r < 1 for r, q in levels):
        raise ValueError('Credibility ceilings must be in [0,1)')
    if any(a[1] < b[1] for a, b in zip(levels, levels[1:])):
        raise ValueError('Planning endpoints must be nonincreasing in the ceiling')
    lower = max([0.] + [r for r, q in levels if q > limit])
    upper = min([1.] + [r for r, q in levels if q <= limit])
    if lower > upper:
        raise ValueError('Inconsistent endpoint decisions')
    return lower, upper

def owa_interval(intervals, slots=None):
    slots = len(intervals) if slots is None else slots
    if slots < len(intervals):
        raise ValueError('Too few cells')
    return (cell_owa([x[0] for x in intervals], slots),
            cell_owa([x[1] for x in intervals], slots))

def grid_width(levels):
    points = sorted(set([0., 1.] + list(levels)))
    return max(b-a for a,b in zip(points, points[1:]))
