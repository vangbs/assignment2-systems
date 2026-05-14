import pickle

with open("data/benchmarking_results.pkl", "rb") as file:
    results = pickle.load(file)

for setup_name, times in results.items():
    print(f"{setup_name}: {times}")