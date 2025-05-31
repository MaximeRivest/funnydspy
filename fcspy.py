# ── funkydspy.py ────────────────────────────────────────────────────────────
"""DSPy, but written like vanilla Python functions.

Supported syntaxes
------------------
1.  Dataclass return + NumPy-style or inline‐comment inputs

        @dataclass
        class Stats:
            mean  : float  = field(metadata={'doc': 'average'})
            above : list[float]

        @fd.funky
        def analyse(nums: list[float], threshold: float) -> Stats:
            "docstring optional"
            return Stats                # body never runs

2.  Dataclass return + *docments* inline comments everywhere

        @dataclass
        class Stats:
            mean  : float      # average
            above : list[float]# > threshold

        @fd.funky
        def analyse(nums: list[float],  # numbers
                     threshold: float): # cut-off
            return Stats

3.  Tuple return annotation, inline comments for inputs, *string
   constants* for output docs

        @fd.funky
        def analyse(nums: list[float], threshold: float) -> tuple[float, list[float]]:
            mean  = "average"
            above = "numbers > threshold"
            return mean, above
"""
from __future__ import annotations
import inspect, ast, textwrap, sys, typing, dataclasses, re, fastcore.docments as fc
import json, functools
import dspy
from dspy import Signature, InputField, OutputField, Example

# ---------- input ► text ---------------------------------------------------
def _to_text(v):
    "Recursively stringify numerics/lists so ChatAdapter never crashes."
    if isinstance(v, list):
        return [_to_text(x) for x in v]
    if isinstance(v, (str, dict)):
        return v
    return str(v)

# ---------- text ► typed ---------------------------------------------------
def _from_text(txt: str, typ):
    "Best-effort cast of `txt` (string) to `typ`."
    origin = typing.get_origin(typ)
    args   = typing.get_args(typ)

    try:
        if typ is float:
            return float(txt)
        if typ is int:
            return int(txt)
        if typ is bool:
            return txt.strip().lower() in ("true", "1", "yes")
        if origin is list and args:
            data = json.loads(txt) if txt.lstrip().startswith("[") else ast.literal_eval(txt)
            return [ _from_text(x, args[0]) for x in data ]
        if origin is dict and args:
            data = json.loads(txt) if txt.lstrip().startswith("{") else ast.literal_eval(txt)
            k_t, v_t = args
            return { _from_text(k, k_t): _from_text(v, v_t) for k,v in data.items() }
    except Exception:
        pass                                 # fall through on failure
    return txt                               # leave as raw string

# ── helpers ────────────────────────────────────────────────────────────────
def _input_descs(fn) -> dict[str,str]:
    "Merge inline *docments* and NumPy Parameters/Inputs."
    return {
        k: v["docment"]
        for k, v in fc.docments(fn, full=True).items()
        if k not in ("return",)
    }

_ATTR = re.compile(r"^\s*([\w_]+)\s*:\s*(.+)$")
def _attrs_from_doc(doc: str):
    lines = doc.splitlines()
    try:
        i = next(j for j,l in enumerate(lines) if l.strip().lower().startswith("attributes"))
    except StopIteration:
        return {}
    out={}
    for l in lines[i+1:]:
        if not l.strip(): break
        m = _ATTR.match(l)
        if m: out[m.group(1)] = m.group(2).strip()
    return out

def _output_specs(fn, sig: inspect.Signature):
    "Return list[(name, type, desc)]"
    ret_ann = sig.return_annotation
    if dataclasses.is_dataclass(ret_ann):
        docmap = _attrs_from_doc(inspect.getdoc(ret_ann) or "")
        out=[]
        for f in dataclasses.fields(ret_ann):
            desc = f.metadata.get("doc") or docmap.get(f.name,"")
            out.append((f.name, f.type, desc))
        return out

    # tuple annotation path – inspect AST for `var = 'desc'` then `return (var, …)`
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    assigns={}
    returns=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                assigns[node.targets[0].id]=node.value.value
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Tuple):
                returns=[elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
            elif isinstance(node.value, ast.Name):
                returns=[node.value.id]
    if returns:
        hints = typing.get_type_hints(fn)
        outs=[]
        for name in returns:
            anno = hints.get(name, str)
            outs.append((name, anno, assigns.get(name,"")))
        return outs

    # fallback single result
    return [("result", str, "")]

# ── @funky decorator ───────────────────────────────────────────────────────
def funky(fn):
    sig_py = inspect.signature(fn)
    in_desc = _input_descs(fn)
    out_spec = _output_specs(fn, sig_py)

    fields={}
    for p in sig_py.parameters:
        fields[p]=InputField(desc=in_desc.get(p,""))
    for n, typ, desc in out_spec:
        fields[n]=OutputField(desc=desc)

    Sig=type(f"{fn.__name__.title()}Sig",(Signature,),fields)

    # default LM is Predict
    default_predict = dspy.Predict(Sig)

    class _Prog:
        signature=Sig
        # normal call → run Predict immediately
        def __call__(self,*a,**k):
            ex = Example(**sig_py.bind_partial(*a,**k).arguments)
            kwargs = {k: _to_text(v) for k, v in dict(ex).items()}
            result = default_predict(**kwargs)
            # Convert outputs back to declared types
            post = {}
            for k,v in dict(result).items():
                field_typ = Sig.output_fields.get(k, None)
                post[k] = _from_text(v, field_typ.annotation if field_typ else str)
            return Example(**post)
        # pipe behaviour → just yield Example
        def __ror__(self,lhs):
            if isinstance(lhs,tuple):
                lhs=dict(zip(sig_py.parameters,lhs))
            if not isinstance(lhs,dict): raise TypeError("lhs must be tuple or dict")
            ex=Example(**lhs); ex._signature=Sig; return ex
        def __repr__(self): return f"<funky {fn.__name__}>"
    return _Prog()

# ── lazy DSPy module wrappers ------------------------------------------------
def _pipe_mod(ModCls):
    class W:
        def __init__(self): self._mods={}
        def __call__(self,*a,**k): return ModCls(*a,**k)
        def __ror__(self,ex:Example):
            sig=getattr(ex,"_signature",None)
            if sig is None: raise ValueError("missing _signature")
            mod=self._mods.get(sig) or ModCls(sig); self._mods[sig]=mod
            kw = ex.toDict() if hasattr(ex,"toDict") else dict(ex)
            kw = {k: _to_text(v) for k, v in kw.items()}
            res=mod(**kw)
            # drop only the inputs; keep outputs *and* reasoning or other extras
            inputs = set(sig.input_fields)
            keep = {k: v for k, v in dict(res).items() if k not in inputs}
            
            # cast outputs back to declared Python types (reasoning stays str)
            typed = {}
            for k, v in keep.items():
                if k in sig.output_fields:
                    field_typ = sig.output_fields[k].annotation
                    typed[k] = _from_text(v, field_typ)
                else:                          # e.g. "reasoning"
                    typed[k] = v
            return Example(**typed)
        def __repr__(self): return f"<pipeable {ModCls.__name__}>"
    return W()

_mod=sys.modules[__name__]
for n,obj in vars(dspy).items():
    if isinstance(obj,type) and issubclass(obj,dspy.Module):
        setattr(_mod,n.lower(),_pipe_mod(obj))
if hasattr(_mod,"chainofthought"):
    setattr(_mod,"cot",getattr(_mod,"chainofthought"))
def register(cls,alias=None): setattr(_mod,alias or cls.__name__.lower(),_pipe_mod(cls))
# ───────────────────────────────────────────────────────────────────────────
