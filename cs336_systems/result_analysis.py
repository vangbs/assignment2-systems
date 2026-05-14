import pickle

with open("data/benchmarking_results.pkl", "rb") as file:
    results = pickle.load(file)

print(results['f'])
print(results['fb'])
print(results['fbs'])