import subprocess as sp


def compile_slang(filename: str) -> str:
    res = sp.run(["slangc", filename, "-target", "wgsl"], capture_output=True)
    if res.returncode != 0 or len(res.stderr) > 0:
        raise Exception(f"Slang compilation error: {res.stderr}")
    source = res.stdout
    return source.decode(encoding="utf8")
