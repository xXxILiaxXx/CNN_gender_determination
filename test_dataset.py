import os
import kagglehub

path = kagglehub.dataset_download("playlist/men-women-classification")
print("PATH:", path)

print("\n== TOP LEVEL ==")
top = os.listdir(path)
print(top)

print("\n== DIRS INSIDE TOP ==")
for name in top:
    p = os.path.join(path, name)
    if os.path.isdir(p):
        print(f"{name}/ -> {os.listdir(p)[:30]}")

# Если есть папка dataset/ или data/ или train/ — покажем глубже на 1 уровень
for candidate in ["dataset", "data", "train", "test", "validation", "valid"]:
    p = os.path.join(path, candidate)
    if os.path.isdir(p):
        print(f"\n== INSIDE {candidate}/ ==")
        print(os.listdir(p)[:50])