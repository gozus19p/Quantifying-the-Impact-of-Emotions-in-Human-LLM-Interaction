from typing import List
from embedding import embed
from vector_db import search
import os
import json


def test() -> None:
    distances = []
    with open(os.path.dirname(__file__) + "/file.txt", "r") as f:

        lines = f.readlines()
        for line in lines:
            print("Testing line {}".format(line))
            results = search(line)
            print(results)
            for result in results:
                distances.append({"distance": result["distance"], "prompt": line})
    with open(os.path.dirname(__file__) + "/distances.json", "w") as f:
        json.dump(distances, f)


test()
