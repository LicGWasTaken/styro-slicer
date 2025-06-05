def nth_char(str_: str, char_: str, n: int):
    all = [i for i, c in enumerate(str_) if c == char_]
    if len(all) <= n:
        return len(str_)
    return all[n]

def nth_substr(str_: str, substr: str, n: int):
    start = str_.find(substr)
    while start >= 0 and n > 1:
        start = str_.find(substr, start+len(substr))
        n -= 1
    return start