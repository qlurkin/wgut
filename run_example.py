import sys
import os

if len(sys.argv) < 2:
    raise Exception("Add the name of an example without the `.py`")

name = sys.argv[1]
name = name + ".py"

examples = os.listdir("./examples")

if name not in examples:
    raise Exception("Not an example")

sys.path.append(os.path.abspath("./src"))

os.chdir("./examples")

with open(name) as file:
    source = file.read()

exec(source)
