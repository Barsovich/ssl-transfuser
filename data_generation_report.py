from pathlib import Path

base_path = Path("./data/")
subdirs = []

for subdir in sorted(list(base_path.iterdir())):
    # print(subdir)
    short_routes = [dir for dir in subdir.iterdir() if "short" in dir.name]
    tiny_routes = [dir for dir in subdir.iterdir() if "tiny" in dir.name]
    # print(f"Short routes: {len(short_routes)}")
    # print(f"Tiny routes: {len(tiny_routes)}")
    # print("\n")
    print(len(short_routes))
    print(len(tiny_routes))
