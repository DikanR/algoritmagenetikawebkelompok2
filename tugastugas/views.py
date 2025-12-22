from django.shortcuts import render
from .algorithms import traveling_salesman_journal as tsp_ag
from typing import Dict
import csv
from django.conf import settings

# def index(request):
#     test = "ini adalah teks"
#     with open("tugastugas/html/tugas.html", "r") as file:
#         bruh = file.read()
#         bruh = bruh.format(test=test)
#         return HttpResponse(bruh)

# def index(request):
#     context = {
#         'test': "ini adalah teks"
#     }
#     return render(request, 'tugas.html', context)

def index(request):
    # cities = {}
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

    cities: Dict[int, Dict[str, float]] = {}

    print(settings.STATICFILES_DIRS[0])

    with open(settings.STATICFILES_DIRS[0] / "world_country_and_usa_states_latitude_and_longitude_values.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            # print(row["latitude"], row["longitude"])
            lat = row["latitude"].strip()
            lon = row["longitude"].strip()

            if not lat or not lon:
                continue  # datanya aneh sumpah

            lat = float(lat)
            lon = float(lon)

            cities[idx] = {
                "lat": lat,
                "lon": lon
            }

    cities_count = len(cities)
    pop_size = 500
    generations = 20
    elite = 8
    tourn_k = 5
    # crossover_rate = 0.0
    # mutation_rate = 0.0
    crossover_rate = 0.6
    mutation_rate = 0.1
    seed = 3

    # elite = 20
    # tourn_k = 7
    # crossover_rate = 0.95
    # mutation_rate = 0.12
    # seed = 42

    lats = []
    lons = []
    if request.method == "POST":
        if request.POST.get('cities_count') is not None and request.POST.get('cities_count') != '':
            cities_count = int(request.POST.get('cities_count'))
            lats = request.POST.getlist('lats[]')
            lons = request.POST.getlist('lons[]')
            cities = {}

            if request.POST.get('generations') is not None and request.POST.get('generations') != '':
                generations = int(request.POST.get('generations'))
            if request.POST.get('crossover_rate') is not None and request.POST.get('crossover_rate') != '':
                crossover_rate = (float(request.POST.get('crossover_rate'))) / 100
            if request.POST.get('mutation_rate') is not None and request.POST.get('mutation_rate') != '':
                mutation_rate = (float(request.POST.get('mutation_rate'))) / 100
    
    if len(cities) == 0:
        for i in range(cities_count):
            if i < len(lats) and i < len(lons):
                cities[i+1] = {
                    "lat": float(lats[i]),
                    "lon": float(lons[i])
                }
            else:
                cities[i+1] = {
                    "lat": 0.0,
                    "lon": 0.0
                }

    # history, best_route, best_dist = tsp_ag.run_ga(
    #     cities=cities,
    #     pop_size=pop_size,
    #     generations=generations,
    #     elite_size=elite,
    #     tournament_k=tourn_k,
    #     crossover_rate=crossover_rate,
    #     mutation_rate=mutation_rate,
    #     seed=seed
    # )

    history, best_route, best_dist, best_cost_history = tsp_ag.run_ga(
        cities=cities,
        pop_size=pop_size,
        generations=generations,
        elite_size=elite,
        tournament_k=tourn_k,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        seed=seed
    )
    img_uri_cities = tsp_ag.plot_cities(cities, show=False)
    img_uri_routes = tsp_ag.plot_route(cities, best_route, best_dist, show=False)
    img_uri_cost_history = tsp_ag.plot_cost_history(best_cost_history, show=False)

    history_last_key = list(history.keys())[-1]
    last_history = {}
    last_history[history_last_key] = history[history_last_key]

    # print(history)
    # print(best_route)
    # print(best_dist)

    context = {
        'cities': cities,
        'cities_count': cities_count,
        'generations': generations,
        'crossover_rate': crossover_rate * 100,
        'mutation_rate': mutation_rate * 100,
        'last_history': last_history,
        'history': history,
        'best_route': best_route,
        'best_dist': best_dist,
        'img_uri_cities': img_uri_cities,
        'img_uri_routes': img_uri_routes,
        'img_uri_cost_history': img_uri_cost_history,
    }
    return render(request, 'traveling_salesman_problem.html', context)