import subprocess as sp
import os
import json
import random
import string


def random_string(length=10):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def search_filename(ext: str):
    found = False
    res = ""
    while found:
        res = random_string(8) + ext
        if not os.path.exists(res):
            found = True
    return res


def compile_slang(filename: str) -> str:
    res = sp.run(["slangc", filename, "-target", "wgsl"], capture_output=True)
    if res.returncode != 0 or len(res.stderr) > 0:
        raise Exception(f"Slang compilation error: {res.stderr}")
    source = res.stdout
    return source.decode(encoding="utf8")


def compile_slang_with_reflexion(filename: str) -> tuple[str, dict]:
    reflection_filename = search_filename(".json")
    res = sp.run(
        [
            "slangc",
            filename,
            "-target",
            "wgsl",
            "-reflection-json",
            reflection_filename,
        ],
        capture_output=True,
    )
    if res.returncode != 0 or len(res.stderr) > 0:
        raise Exception(f"Slang compilation error: {res.stderr}")
    source = res.stdout
    reflection = {}
    with open(reflection_filename) as file:
        reflection = json.load(file)
    os.remove(reflection_filename)
    return source.decode(encoding="utf8"), reflection
