input_path = "../data/FIN5_fixed.txt"
train_path = "../data/FIN5_train.txt"
dev_path = "../data/FIN5_dev.txt"

contracts = []
current = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 1 and parts[1] == "-DOCSTART-":
            if current:
                contracts.append(current)
                current = []
        current.append(line)
    if current:
        contracts.append(current)

# Sanity check
if len(contracts) < 5:
    print(f"Warning: Only found {len(contracts)} contracts!")

# Split: first 4 for train, last 1 for dev
train_contracts = contracts[:4]
dev_contracts = contracts[4:]

with open(train_path, "w", encoding="utf-8") as f:
    for contract in train_contracts:
        f.writelines(contract)

with open(dev_path, "w", encoding="utf-8") as f:
    for contract in dev_contracts:
        f.writelines(contract)

print(f"Train contracts: {len(train_contracts)}, Dev contracts: {len(dev_contracts)}")