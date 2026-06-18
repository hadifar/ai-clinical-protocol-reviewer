from services.rank import search

results = search("primary study objectives")
for r in results:
    print(r["original"])
    print('-'*100)
