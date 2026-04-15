# ruff-ignored garbage + type error + bad import
import nonexistent_zzz  # unresolved

def add(a, b):  # missing type hints -> mypy --strict flags this
    x = 1;  # ruff E702
    return a + b + nonexistent_zzz.unknown()
