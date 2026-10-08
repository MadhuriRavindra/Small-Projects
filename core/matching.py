"""Match free text / filenames to products in the catalog (used by PDF, web and Reddit loaders)."""
import re

_CODE = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9]{5,}$")  # model codes like 15arp9, 15iah8, 7730u


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def aliases_for(brand: str, model: str) -> set[str]:
    m, b = norm(model), norm(brand)
    out = {m, f"{b} {m}"}
    toks = m.split()
    if len(toks) > 2 and _CODE.match(toks[-1]):  # "legion 5 15arp9" -> also "legion 5"
        short = " ".join(toks[:-1])
        out.update({short, f"{b} {short}"})
    return out


def build_alias_index(catalog: list[dict]) -> list[tuple[str, str]]:
    """[(alias, product_id)], longest alias first."""
    idx = [(a, c["product_id"]) for c in catalog for a in aliases_for(c["brand"], c["model"])]
    return sorted(idx, key=lambda x: -len(x[0]))


def match_text(text: str, alias_index: list[tuple[str, str]]) -> set[str]:
    """Product ids mentioned in text. A match is dropped if a longer matching alias of another
    product contains it (so 'macbook pro 14 m3 pro' does not also count as 'macbook pro 14 m3')."""
    t = f" {norm(text)} "
    found = [(a, pid) for a, pid in alias_index if f" {a} " in t]
    keep = set()
    for a, pid in found:
        if any(a != a2 and a in a2 and pid != pid2 for a2, pid2 in found):
            continue
        keep.add(pid)
    return keep
