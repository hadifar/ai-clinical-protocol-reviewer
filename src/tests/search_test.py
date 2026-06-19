from services.rank import search

results = search("primary study objectives")
for r in results:
    print(r["score"])
    print("-" * 100)
