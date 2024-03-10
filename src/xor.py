
import random
from typing import Iterable

num_features = 8
num_samples = 10000

# implements XOR
def one(itr: Iterable[object]):
    found_one: bool = False
    for obj in itr:
        if bool(obj) == True:
            if found_one:
                return False
            else:
                found_one = True
                
    return found_one

            


i = 0
while i < num_samples/2:
    features = [random.randint(0,1) for _ in range(num_features)]
    label = int(one(features))
    if label == 0:
        continue
    else:
        i += 1


    for f in features:
        print(f"{f},", end='')
    print(label)


i = 0
while i < num_samples/2:
    features = [random.randint(0,1) for _ in range(num_features)]
    label = int(one(features))
    if label == 1:
        continue
    else:
        i += 1


    for f in features:
        print(f"{f},", end='')
    print(label)


