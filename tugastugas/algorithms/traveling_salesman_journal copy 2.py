import random
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import io
import base64

# --------------------------
# Cities as numeric keys (1..15)
# Coordinates from Table III of the uploaded journal.
# --------------------------
# cities: Dict[int, Dict[str, float]] = {
#     1:  {"lat": -27.6126, "lon": -51.0233},
#     2:  {"lat": -26.5716, "lon": -52.3229},
#     3:  {"lat": -27.4087, "lon": -49.8220},
#     4:  {"lat": -27.2662, "lon": -49.7080},
#     5:  {"lat": -26.9985, "lon": -51.5528},
#     6:  {"lat": -27.0754, "lon": -52.9808},
#     7:  {"lat": -26.8794, "lon": -52.8568},
#     8:  {"lat": -27.6963, "lon": -48.8243},
#     9:  {"lat": -26.8430, "lon": -53.5758},
#     10: {"lat": -26.7810, "lon": -49.3593},
#     11: {"lat": -27.4960, "lon": -48.6598},
#     12: {"lat": -26.9155, "lon": -49.0709},
#     13: {"lat": -27.7455, "lon": -49.9423},
#     14: {"lat": -28.3377, "lon": -49.6373},
#     15: {"lat": -26.7326, "lon": -52.3919},
# }

# city_keys: List[int] = sorted(cities.keys())  # [1,2,...,15]

# --------------------------
# Distance: SIMPLE PLANAR (km)
# --------------------------
def planar_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    km_per_deg = 111.0  # rough conversion: 1 degree ~ 111 km (approx.)
    dx = (lat2 - lat1) * km_per_deg
    dy = (lon2 - lon1) * km_per_deg
    return math.hypot(dx, dy)

# Precompute distance dictionary dist[i][j]
# dist: Dict[int, Dict[int, float]] = {}
# for i in city_keys:
#     dist[i] = {}
#     for j in city_keys:
#         if i == j:
#             dist[i][j] = 0.0
#         else:
#             dist[i][j] = haversine_km(cities[i]["lat"], cities[i]["lon"],
#                                       cities[j]["lat"], cities[j]["lon"])

# --------------------------
# GA primitives (operate on routes that are lists of numeric city keys)
# --------------------------
def route_length(route: List[int], dist: Dict[int, Dict[int, float]] = {}) -> float:
    total = 0.0
    n = len(route)
    for k in range(n):
        a = route[k]
        b = route[(k + 1) % n]  # close the tour
        total += dist[a][b]
    return total

def random_route(city_keys) -> List[int]:
    r = city_keys[:]
    random.shuffle(r)
    return r

def ordered_crossover(p1: List[int], p2: List[int]) -> List[int]:
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # copy segment from p1
    child[a:b+1] = p1[a:b+1]
    # fill with genes from p2 in order
    pos = (b + 1) % size
    idx = (b + 1) % size
    while None in child:
        gene = p2[idx]
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
        idx = (idx + 1) % size
    return child

def swap_mutation(route: List[int], mutation_rate: float) -> None:
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]

def tournament_selection(pop: List[List[int]], fitnesses: List[float], k: int) -> List[int]:
    chosen = random.sample(range(len(pop)), k)
    best_idx = min(chosen, key=lambda idx: fitnesses[idx])  # minimize distance
    return pop[best_idx][:]  # return a copy

# --------------------------
# GA main routine
# --------------------------
def run_ga(
    cities: Dict[int, Dict[str, float]] = {},
    pop_size: int = 300,
    generations: int = 500,
    elite_size: int = 10,
    tournament_k: int = 5,
    crossover_rate: float = 0.95,
    mutation_rate: float = 0.08,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[Dict[int, Dict[str, object]], List[int], float]:
    random.seed(seed)

    # build distance matrix here using planar metric (function name unchanged)
    city_keys_local: List[int] = sorted(cities.keys())
    dist_local: Dict[int, Dict[int, float]] = {}
    for i in city_keys_local:
        dist_local[i] = {}
        for j in city_keys_local:
            if i == j:
                dist_local[i][j] = 0.0
            else:
                dist_local[i][j] = planar_km(cities[i]["lat"], cities[i]["lon"],
                                                cities[j]["lat"], cities[j]["lon"])

    n = len(city_keys_local)
    population = [random_route(city_keys_local) for _ in range(pop_size)]
    fitnesses = [route_length(r, dist_local) for r in population]

    history = {}
    best_dist = min(fitnesses)
    best_route = population[fitnesses.index(best_dist)][:]

    for gen in range(1, generations + 1):
        # elitism: keep top elite_size
        zipped = list(zip(population, fitnesses))
        zipped.sort(key=lambda x: x[1])
        new_pop = [ind[:] for ind, _ in zipped[:elite_size]]

        # fill rest
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, tournament_k)
            p2 = tournament_selection(population, fitnesses, tournament_k)
            if random.random() < crossover_rate:
                child = ordered_crossover(p1, p2)
            else:
                child = p1[:]
            swap_mutation(child, mutation_rate)
            new_pop.append(child)

        population = new_pop
        fitnesses = [route_length(r, dist_local) for r in population]

        gen_best = min(fitnesses)
        if gen_best < best_dist:
            best_dist = gen_best
            best_route = population[fitnesses.index(gen_best)][:]

        if verbose and (gen % max(1, generations // 10) == 0 or gen in (1, generations)):
            history[gen] = {'best_dist': best_dist, 'best_route': best_route}
            # print(f"Gen {gen:4d} | Best distance: {best_dist:.3f} km")

    return history, best_route, best_dist

# --------------------------
# Plotting
# --------------------------
def _fig_to_base64(fig) -> str:
    """Helper: convert current matplotlib figure to a PNG base64 data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return "data:image/png;base64," + b64

def plot_cities(cities: Dict[int, Dict[str, float]], show: bool = True) -> str:
    """
    Plot cities (x = latitude, y = longitude) and return a PNG base64 data URI.
    Returns: data URI string 'data:image/png;base64,...'
    """
    city_keys_local: List[int] = sorted(cities.keys())
    lats = [cities[k]["lat"] for k in city_keys_local]
    lons = [cities[k]["lon"] for k in city_keys_local]
    labels = [str(k) for k in city_keys_local]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(lats, lons)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (lats[i], lons[i]), textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_title("Cities")
    ax.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()

    data_uri = _fig_to_base64(fig)

    if show:
        plt.show()
    plt.close(fig)
    return data_uri

def plot_route(cities: Dict[int, Dict[str, float]], route: List[int], total_distance: float,
               show: bool = True) -> str:
    """
    Plot route (x = latitude, y = longitude) and return a PNG base64 data URI.
    Returns: data URI string 'data:image/png;base64,...'
    """
    ordered_lats = [cities[c]["lat"] for c in route] + [cities[route[0]]["lat"]]
    ordered_lons = [cities[c]["lon"] for c in route] + [cities[route[0]]["lon"]]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ordered_lats, ordered_lons, marker="o", linewidth=1)
    for idx, city in enumerate(route):
        label = f"{idx+1}:{city}"
        ax.annotate(label, (cities[city]["lat"], cities[city]["lon"]), textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_title(f"Jarak Terpendek\nTotal distance: {total_distance:.3f} km")
    ax.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()

    data_uri = _fig_to_base64(fig)

    if show:
        plt.show()
    plt.close(fig)
    return data_uri
