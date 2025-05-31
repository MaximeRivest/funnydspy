# ── funkydspy.py ────────────────────────────────────────────────────────────
import inspect, typing, dataclasses, sys, textwrap, re
import dspy
from dspy import Signature, Example, InputField, OutputField

# 1. Helper to wrap any DSPy Module class in a lazy, pipe-friendly shell
def _lazy_factory(ModuleCls):
    class LazyPipe:
        def __init__(self, *init_args, **init_kw):
            self._init_args, self._init_kw = init_args, init_kw
            self._modules: dict[Signature, dspy.Module] = {}   # <─ NEW cache per signature
        def __call__(self, sig):
            return ModuleCls(sig, *self._init_args, **self._init_kw)
        def __ror__(self, example: Example):
            sig = getattr(example, "_signature", None)
            if sig is None:
                raise ValueError("Upstream object has no _signature; cannot auto-instantiate DSPy module.")
            
            # ── get or build the module tied to *this* Signature ──────────
            module = self._modules.get(sig)
            if module is None:
                module = ModuleCls(sig, *self._init_args, **self._init_kw)
                self._modules[sig] = module
            
            if hasattr(example, "toDict"):
                kwargs = example.toDict()
            else:
                kwargs = dict(example)
            def to_text(v):
                if isinstance(v, list):
                    return [to_text(x) for x in v]
                if isinstance(v, (str, dict)):
                    return v
                return str(v)
            kwargs = {k: to_text(v) for k, v in kwargs.items()}
            result = module(**kwargs)
            # --- Only keep outputs ---
            declared = set(sig.output_fields)   # <---- use signature from cache key
            cleaned  = {k: v for k, v in dict(result).items() if k in declared}
            return Example(**cleaned)
        def __repr__(self):
            return f"<lazy {ModuleCls.__name__}>"
    return LazyPipe

# 2. Helpers to parse docstring for input/output field descs
_SECTION = re.compile(r"^[A-Z][A-Za-z0-9 _]+$")       # Section headers
_FIELD   = re.compile(r"^\s*([\w_]+)\s*:\s*([^\n]+)$") #   name : text blob

def _parse_section(doc: str, section_name: str) -> dict[str, str]:
    lines = doc.splitlines()
    start = next(
        (i for i, l in enumerate(lines)
         if l.strip().lower().startswith(section_name.lower())),
        None,
    )
    if start is None:
        return {}
    out = {}
    for l in lines[start + 1:]:
        if not l.strip() or _SECTION.match(l.strip()):
            break
        m = _FIELD.match(l)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out

def _input_descs(doc: str) -> dict[str, str]:
    desc = _parse_section(doc, "inputs")
    if not desc:
        desc = _parse_section(doc, "parameters")
    return desc

# 3. Output field parser (outputs only from Returns or annotation)
_RET_LINE   = re.compile(r"^\s*([\w_]+)\s*:\s*([^\n]+)$")
_SECTION_RE = re.compile(r"^[A-Z][A-Za-z0-9 _]+$")

def _output_fields(ret_ann, fn):
    # Handle inspect._empty explicitly
    if ret_ann is inspect.Parameter.empty:
        ret_ann = typing.Any
        
    if dataclasses.is_dataclass(ret_ann):
        return [(f.name, f.type) for f in dataclasses.fields(ret_ann)]
    if isinstance(ret_ann, type) and issubclass(ret_ann, tuple) and hasattr(ret_ann, "_fields"):
        anns = ret_ann.__annotations__
        return [(n, anns.get(n, typing.Any)) for n in ret_ann._fields]
    if isinstance(ret_ann, type) and issubclass(ret_ann, dict) and hasattr(ret_ann, "__annotations__"):
        return list(ret_ann.__annotations__.items())
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().lower().startswith("returns")), None)
    if start is not None:
        collected = []
        for l in lines[start+1:]:
            if not l.strip():
                break
            if _SECTION_RE.match(l.strip()):
                break
            m = _RET_LINE.match(l)
            if m:
                collected.append((m.group(1), m.group(2)))
        if collected:
            return collected
    return [("result", ret_ann or typing.Any)]

# 4. The @funky decorator
def funky(fn):
    sig_py = inspect.signature(fn)
    sig_fields = {}
    doc = inspect.getdoc(fn) or ""
    idesc = _input_descs(doc)
    # Inputs with descs
    for name, _ in sig_py.parameters.items():
        sig_fields[name] = InputField(desc=idesc.get(name, ""))
    # Outputs with descs/types
    for name, typ in _output_fields(sig_py.return_annotation, fn):
        sig_fields[name] = OutputField(desc=str(typ))
    Sig = type(f"{fn.__name__.title()}Sig", (Signature,), sig_fields)
    class _Prog:
        signature = Sig
        def __call__(self, *a, **k):
            bound = sig_py.bind_partial(*a, **k)
            ex = Example(**bound.arguments)
            ex._signature = Sig
            return ex
        def __ror__(self, lhs):
            if isinstance(lhs, tuple):
                if len(lhs) != len(sig_py.parameters):
                    raise TypeError("Tuple length mismatch")
                lhs = dict(zip(sig_py.parameters, lhs))
            if not isinstance(lhs, dict):
                raise TypeError("lhs of | must be tuple or dict")
            ex = Example(**lhs)
            ex._signature = Sig
            return ex
        def __repr__(self):
            return f"<funky {fn.__name__}>"
    return _Prog()

# 5. Expose lazy factories for *all* DSPy Modules (predict, cot, react, …)
_this = sys.modules[__name__]
def _auto_register():
    for name, obj in vars(dspy).items():
        if isinstance(obj, type) and issubclass(obj, dspy.Module):
            alias = name.lower()
            setattr(_this, alias, _lazy_factory(obj)())
    if hasattr(_this, "chainofthought"):
        setattr(_this, "cot", getattr(_this, "chainofthought"))
_auto_register()

def register(module_cls, alias=None):
    alias = alias or module_cls.__name__.lower()
    setattr(_this, alias, _lazy_factory(module_cls))

# ── end funkydspy.py ────────────────────────────────────────────────────────
