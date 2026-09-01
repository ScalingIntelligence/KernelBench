from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


Config = Dict[str, Any]
DomainFn = Callable[[Config], Iterable[Any]]
Domain = Union[Iterable[Any], DomainFn]
CaseKey = TypeVar("CaseKey", bound=Hashable)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    domain: Optional[Domain] = None
    default: Any = None
    meaningful_if: Callable[[Config], bool] = lambda cfg: True
    compute: Optional[Callable[[Config], Any]] = None
    depends_on: Tuple[str, ...] = ()
    sweep: bool = True

    export: bool = True


class KernelConfigGraph:
    def __init__(self, specs: List[ParamSpec]):
        self.specs: Dict[str, ParamSpec] = {s.name: s for s in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Duplicate param names in specs.")
        self._order = self._toposort_or_raise()

    def dependency_edges(self) -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for v, spec in self.specs.items():
            for u in spec.depends_on:
                if u not in self.specs:
                    raise KeyError(f"Param '{v}' depends on unknown param '{u}'.")
                edges.append((u, v))
        return edges

    def _toposort_or_raise(self) -> List[str]:
        edges = self.dependency_edges()
        outgoing: Dict[str, Set[str]] = {k: set() for k in self.specs}
        indeg: Dict[str, int] = {k: 0 for k in self.specs}

        for u, v in edges:
            if v not in outgoing[u]:
                outgoing[u].add(v)
                indeg[v] += 1

        q = [k for k, d in indeg.items() if d == 0]
        order: List[str] = []
        while q:
            n = q.pop()
            order.append(n)
            for m in list(outgoing[n]):
                outgoing[n].remove(m)
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)

        if len(order) != len(self.specs):
            cyc = [k for k, d in indeg.items() if d > 0]
            raise ValueError(f"Dependency cycle detected among: {cyc}")
        return order

    def _get_domain_values(self, spec: ParamSpec, cfg: Config) -> List[Any]:
        if spec.domain is None:
            return []
        dom = spec.domain
        vals = dom(cfg) if callable(dom) else dom
        return list(vals)

    def _strip_non_exported(self, cfg: Config) -> Config:
        return {k: v for k, v in cfg.items() if self.specs[k].export}

    def resolve_and_expand(self) -> List[Config]:
        configs: List[Config] = [dict()]

        for name in self._order:
            spec = self.specs[name]
            next_configs: List[Config] = []

            for cfg0 in configs:
                cfg = dict(cfg0)

                if name not in cfg:
                    cfg[name] = spec.default

                if not spec.meaningful_if(cfg):
                    cfg[name] = spec.default
                    next_configs.append(cfg)
                    continue

                if spec.compute is not None:
                    cfg[name] = spec.compute(cfg)
                    next_configs.append(cfg)
                    continue

                if spec.sweep and spec.domain is not None:
                    vals = self._get_domain_values(spec, cfg)
                    if len(vals) == 0:
                        cfg[name] = spec.default
                        next_configs.append(cfg)
                    else:
                        for v in vals:
                            cfg_branch = dict(cfg)
                            cfg_branch[name] = v
                            next_configs.append(cfg_branch)
                    continue

                next_configs.append(cfg)

            configs = next_configs

        # Strip non-exported keys, then dedup
        seen: Set[Tuple[Tuple[str, Any], ...]] = set()
        out: List[Config] = []
        for c in configs:
            c2 = self._strip_non_exported(c)
            key = tuple(sorted(c2.items()))
            if key not in seen:
                seen.add(key)
                c2_sorted = dict(sorted(c2.items(), key=lambda kv: kv[0]))
                out.append(c2_sorted)
        return out


def domain_by_case(
    case: Callable[[Config], CaseKey],
    table: Mapping[CaseKey, Mapping[str, Any]],
    key: str,
    *,
    empty_ok: bool = False,
) -> Callable[[Config], Iterable[Any]]:
    def _domain(cfg: Config) -> Iterable[Any]:
        c = case(cfg)
        if c not in table or key not in table[c]:
            if empty_ok:
                return []
            raise KeyError(f"Missing domain for key='{key}' under case='{c}'")
        val = table[c][key]
        # Require list-like for domain
        if not isinstance(val, (list, tuple)):
            raise TypeError(
                f"CASE_TABLE[{c}][{key}] must be list/tuple for a domain, got {type(val)}"
            )
        return val

    return _domain


def value_by_case(
    case: Callable[[Config], CaseKey],
    table: Mapping[CaseKey, Mapping[str, Any]],
    key: str,
    *,
    required: bool = True,
) -> Callable[[Config], Any]:
    def _compute(cfg: Config) -> Any:
        c = case(cfg)
        if c in table and key in table[c]:
            return table[c][key]
        if required:
            raise KeyError(f"Missing value for key='{key}' under case='{c}'")
        return None

    return _compute
