# Read weights from file
with open('weights.txt', 'r') as f:
    weights = []
    for line in f:
        weights.append([float(x) for x in line.split()])


print(weights)