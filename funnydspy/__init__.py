from __future__ import annotations
"""funnydspy — vanilla-Python ergonomics on top of DSPy

Write plain functions/dataclasses and get a DSPy module *and* normal Python
return values.  If you need the original DSPy ``Prediction`` (for optimisation or
loss computation) just call the function with ``_prediction=True``::

    stats   = analyse(nums, thr)                  # → Stats dataclass
    pred    = analyse(nums, thr, _prediction=True)  # → dspy.Prediction

Pipe usage (``lhs | analyse``) still returns a ``dspy.Example`` so DSPy chains
and optimisers keep working.

⚠️  CHANGELOG (2025-06-01)
-------------------------
* **Dataclass-aware output naming** - return-type fields are now exported as
  ``<DataclassName>_<field>`` to exactly match the convention used by vanilla
  DSPy.  Example: ``Stats.mean_value`` → output field ``Stats_mean_value``.
* **Smarter reconstruction** - the call wrapper now strips the dataclass
  prefix when rebuilding the structured return object so existing user code
  (expecting a plain ``Stats`` instance) continues to work unchanged.
"""

__version__ = "0.2.0"
__author__ = "FunnyDSPy Contributors"
__email__ = ""
__description__ = "Vanilla-Python ergonomics on top of DSPy"

import inspect, ast, textwrap, sys, typing, dataclasses, re, json
from typing import Any
import fastcore.docments as fc
import dspy
from dspy import Signature, InputField, OutputField, Example, Prediction

# ──────────────────────────────────────────────────────────────────────────────
# utils: serialise → LM-safe strings
# -----------------------------------------------------------------------------

def _to_text(v: Any):
    """Recursively cast numerics/lists to ``str`` so ChatAdapter never crashes."""
    if isinstance(v, list):
        return [_to_text(x) for x in v]
    if isinstance(v, (str, dict)):
        return v
    return str(v)

# helpers for pulling docstrings / inline comments
# -----------------------------------------------------------------------------

def _input_descs(fn) -> dict[str, str]:
    """Merge inline *docments* and NumPy-style *Parameters* section."""
    try:
        return {
            k: v["docment"]
            for k, v in fc.docments(fn, full=True).items()
            if k != "return"
        }
    except (OSError, TypeError, AttributeError):
        # Handle cases where source code cannot be retrieved (e.g., interactive shell)
        return {}

_ATTR = re.compile(r"^\s*([\w_]+)\s*:\s*(.+)$")

def _attrs_from_doc(doc: str):
    lines = doc.splitlines()
    try:
        i = next(j for j, l in enumerate(lines) if l.strip().lower().startswith("attributes"))
    except StopIteration:
        return {}
    out: dict[str, str] = {}
    for l in lines[i + 1 :]:
        if not l.strip():
            break
        m = _ATTR.match(l)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out

def _extract_inline_comments(fn) -> dict[str, str]:
    """Extract inline comments from dataclass fields."""
    try:
        source = inspect.getsource(fn)
        lines = source.splitlines()
        comments = {}
        
        for line in lines:
            # Look for patterns like: field_name: Type # comment
            if '#' in line and ':' in line:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    field_part = parts[0].strip()
                    comment = parts[1].strip()
                    
                    # Extract field name from "field_name: Type" pattern
                    if ':' in field_part:
                        field_name = field_part.split(':')[0].strip()
                        if field_name and not field_name.startswith('@') and not field_name.startswith('def'):
                            comments[field_name] = comment
        return comments
    except (OSError, TypeError, AttributeError):
        # Handle cases where source code cannot be retrieved (e.g., interactive shell)
        return {}

def _extract_dataclass_comments(dataclass_type) -> dict[str, str]:
    """Extract inline comments from dataclass field definitions."""
    try:
        source = inspect.getsource(dataclass_type)
        lines = source.splitlines()
        comments = {}
        
        for line in lines:
            # Look for patterns like: field_name: Type # comment
            if '#' in line and ':' in line:
                parts = line.split('#', 1)
                if len(parts) == 2:
                    field_part = parts[0].strip()
                    comment = parts[1].strip()
                    
                    # Extract field name from "field_name: Type" pattern
                    if ':' in field_part:
                        field_name = field_part.split(':')[0].strip()
                        if field_name and not field_name.startswith('@') and not field_name.startswith('class'):
                            comments[field_name] = comment
        return comments
    except (OSError, TypeError, AttributeError):
        # Handle cases where source code cannot be retrieved (e.g., interactive shell)
        return {}

# utils: cast LM string → declared Python type
# -----------------------------------------------------------------------------

def _from_text(txt: str, typ):
    """Best-effort cast of *txt* (string) to *typ*."""
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
            # Handle both JSON format and simple comma-separated values
            if txt.strip().startswith("[") and txt.strip().endswith("]"):
                data = json.loads(txt)
            else:
                # Try to parse as comma-separated values
                try:
                    data = ast.literal_eval(txt) if txt.strip().startswith("[") else [x.strip() for x in txt.split(",")]
                except:
                    data = [x.strip() for x in txt.split(",")]
            return [_from_text(str(x), args[0]) for x in data]
        if origin is dict and args:
            data = json.loads(txt) if txt.lstrip().startswith("{") else ast.literal_eval(txt)
            k_t, v_t = args
            return {_from_text(k, k_t): _from_text(v, v_t) for k, v in data.items()}
    except Exception:
        pass  # fall through on failure
    return txt  # raw string

# -----------------------------------------------------------------------------
# return-type introspection → list[(name, type, desc, raw_name)]
# -----------------------------------------------------------------------------

def _output_specs(fn, sig: inspect.Signature):
    """Return a list describing each *output* field.

    Elements are ``(field_name_in_signature, annotated_type, description,
    original_attr_name)``.  ``field_name_in_signature`` follows DSPy naming
    convention (e.g. ``Stats_mean_value``) while ``original_attr_name`` keeps
    the raw dataclass field for later reconstruction.
    """
    ret_ann = sig.return_annotation

    # 1️⃣ external @dataclass --------------------------------------------------
    if dataclasses.is_dataclass(ret_ann):
        docmap = _attrs_from_doc(inspect.getdoc(ret_ann) or "")
        inline_comments = _extract_inline_comments(fn)
        dataclass_comments = _extract_dataclass_comments(ret_ann)
        pref   = f"{ret_ann.__name__}_"
        return [(
            f"{pref}{f.name}",        # field name with prefix
            f.type,                    # annotated type
            f.metadata.get("doc") or docmap.get(f.name, "") or inline_comments.get(f.name, "") or dataclass_comments.get(f.name, ""),
            f.name                     # raw field name (no prefix)
        ) for f in dataclasses.fields(ret_ann)]

    # 2️⃣ external NamedTuple --------------------------------------------------
    if isinstance(ret_ann, type) and issubclass(ret_ann, tuple):
        hints = ret_ann.__annotations__
        inline_comments = _extract_inline_comments(fn)
        return [(k, hints[k], inline_comments.get(k, ""), k) for k in hints]

    # 3️⃣ internal class returned directly ------------------------------------
    try:
        src  = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
        
        # Find all class definitions (including nested ones)
        class_defs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_defs[node.name] = node

        cls_name: str | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    # Found return ClassName(...) pattern
                    cls_name = node.value.func.id
                    break
                elif isinstance(node.value, ast.Name):
                    # Found return ClassName pattern
                    cls_name = node.value.id
                    break

        if cls_name and cls_name in class_defs:
            cls_node = class_defs[cls_name]
            
            # Extract field annotations from the class definition
            field_annotations = {}
            field_comments = {}
            
            for node in cls_node.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    # Found field annotation: field_name: Type
                    field_name = node.target.id
                    field_annotations[field_name] = node.annotation
                    
            # Try to extract inline comments from the source
            try:
                class_source_lines = src.splitlines()
                for line in class_source_lines:
                    if '#' in line and ':' in line:
                        parts = line.split('#', 1)
                        if len(parts) == 2:
                            field_part = parts[0].strip()
                            comment = parts[1].strip()
                            if ':' in field_part:
                                field_name = field_part.split(':')[0].strip()
                                if field_name in field_annotations:
                                    field_comments[field_name] = comment
            except:
                pass
            
            if field_annotations:
                # Convert AST annotations to actual types and return field specifications
                converted_annotations = []
                for field_name in field_annotations:
                    ast_annotation = field_annotations[field_name]
                    try:
                        # Try to evaluate the AST annotation to get the actual type
                        # First, get the annotation as a string
                        if hasattr(ast, 'unparse'):
                            type_str = ast.unparse(ast_annotation)
                        else:
                            # Fallback for older Python versions
                            type_str = str(ast_annotation)
                        
                        # Try to evaluate it in the function's context
                        func_globals = fn.__globals__ if hasattr(fn, '__globals__') else {}
                        func_locals = {'typing': typing, 'List': typing.List, 'list': list, 'float': float, 'int': int, 'str': str}
                        
                        try:
                            actual_type = eval(type_str, func_globals, func_locals)
                        except:
                            # If evaluation fails, use the string representation
                            actual_type = str
                        
                        converted_annotations.append((
                            field_name, 
                            actual_type, 
                            field_comments.get(field_name, ""), 
                            field_name
                        ))
                    except:
                        # Fallback to string type if conversion fails
                        converted_annotations.append((
                            field_name, 
                            str, 
                            field_comments.get(field_name, ""), 
                            field_name
                        ))
                
                return converted_annotations
            
            # Fallback: try to compile and execute the class
            try:
                compiled = compile(ast.Module(body=[cls_node], type_ignores=[]), "<inner_class>", "exec")
                scope: dict[str, Any] = {}
                exec(compiled, scope)
                cls_obj = scope[cls_name]
                if dataclasses.is_dataclass(cls_obj):
                    pref = f"{cls_obj.__name__}_"
                    inline_comments = _extract_inline_comments(fn)
                    return [(f"{pref}{f.name}", f.type, inline_comments.get(f.name, ""), f.name) for f in dataclasses.fields(cls_obj)]
                if issubclass(cls_obj, tuple) and hasattr(cls_obj, '__annotations__'):
                    # This is a NamedTuple - extract field names and comments
                    hints = cls_obj.__annotations__
                    return [(k, hints[k], field_comments.get(k, ""), k) for k in hints]
            except:
                pass
    except (OSError, TypeError, AttributeError, SyntaxError):
        # Handle cases where source code cannot be retrieved or parsed
        pass

    # 4️⃣ tuple[...] or primitive fallback ------------------------------------
    if typing.get_origin(ret_ann) is tuple:
        args = typing.get_args(ret_ann) or (str,)
        
        # Try to extract variable names from return statement
        var_names = _extract_return_variable_names(fn)
        var_descriptions = _extract_variable_descriptions(fn)
        
        if var_names and len(var_names) == len(args):
            # Use extracted variable names
            names = var_names
            descriptions = [var_descriptions.get(name, "") for name in names]
        else:
            # Fallback to generic names
            names = [f"field{i}" for i in range(len(args))]
            descriptions = [""] * len(args)
            
        return list(zip(names, args, descriptions, names))

    # Single return value - try to extract variable name
    var_names = _extract_return_variable_names(fn)
    if var_names and len(var_names) == 1:
        var_name = var_names[0]
        var_descriptions = _extract_variable_descriptions(fn)
        description = var_descriptions.get(var_name, "")
        return [(var_name, ret_ann if ret_ann is not inspect._empty else str, description, var_name)]
    
    return [("result", ret_ann if ret_ann is not inspect._empty else str, "", "result")]

# -----------------------------------------------------------------------------
# core decorator
# -----------------------------------------------------------------------------

def funky(fn=None, *, ModCls: type[dspy.Module] = dspy.Predict):
    if fn is None:
        return lambda f: funky(f, ModCls=ModCls)

    sig_py   = inspect.signature(fn)
    in_desc  = _input_descs(fn)
    out_spec = _output_specs(fn, sig_py)

    # Signature subclass ------------------------------------------------------
    fields: dict[str, Any] = {}
    annotations: dict[str, Any] = {}
    
    for p in sig_py.parameters:  # inputs
        param = sig_py.parameters[p]
        fields[p] = InputField(desc=in_desc.get(p, ""))
        annotations[p] = param.annotation if param.annotation != inspect.Parameter.empty else str
        
    for n, typ, desc, _ in out_spec:  # outputs
        fields[n] = OutputField(desc=desc)
        annotations[n] = typ

    # Create signature class with proper annotations
    class_dict = fields.copy()
    class_dict['__annotations__'] = annotations
    if fn.__doc__:
        class_dict['__doc__'] = fn.__doc__
    
    Sig = type(f"{fn.__name__.title()}Sig", (Signature,), class_dict)
    default_mod = ModCls(Sig)

    # module wrapper ----------------------------------------------------------
    class _Prog:
        """Wrapper around the underlying DSPy module (``default_mod``).

        * Direct call → native Python or ``dspy.Prediction``
          (if ``_prediction=True``).
        * Pipe (``lhs | prog``) → ``dspy.Example`` for chaining.
        * ``prog.module`` (alias ``prog._dspy``) → raw DSPy module.
        """
        signature = Sig

        def __call__(self, *a, _prediction: bool = False, **k):
            if "_prediction" in k:
                raise TypeError("pass _prediction without the preceding * in positional/keyword mix")
            ex = Example(**sig_py.bind_partial(*a, **k).arguments)
            kwargs = {kk: _to_text(vv) for kk, vv in dict(ex).items()}
            res: Prediction = default_mod(**kwargs)
            if _prediction:
                return res

            # cast outputs → Python types -----------------------------------
            post: dict[str, Any] = {}
            ret_ann = sig_py.return_annotation

            if dataclasses.is_dataclass(ret_ann):
                pref = f"{ret_ann.__name__}_"
                for kk, vv in dict(res).items():
                    if kk.startswith(pref):
                        raw = kk[len(pref):]  # strip dataclass prefix
                        ann = Sig.output_fields[kk].annotation
                        post[raw] = _from_text(vv, ann)
                return ret_ann(**post)  # Stats(**...)

            # other structured returns -------------------------------------
            for kk, vv in dict(res).items():
                if kk in Sig.output_fields:  # Only process fields we defined
                    ann = Sig.output_fields[kk].annotation
                    post[kk] = _from_text(vv, ann)

            if isinstance(ret_ann, type) and issubclass(ret_ann, tuple) and hasattr(ret_ann, "_fields"):
                return ret_ann(*[post[n] for n in ret_ann._fields])
            
            # Check if this is an internal NamedTuple case
            # This happens when the function defines a NamedTuple inside and returns it
            if typing.get_origin(ret_ann) is tuple and len(post) > 1:
                # Try to detect if we have NamedTuple field names
                field_names = list(post.keys())
                # Check if field names look like NamedTuple fields (not generic field0, field1)
                if (len(field_names) > 1 and 
                    all(not name.startswith('field') for name in field_names) and
                    all(name.isidentifier() for name in field_names)):
                    # This looks like a NamedTuple - create a dynamic NamedTuple class
                    from typing import NamedTuple
                    from collections import namedtuple
                    
                    # Create a NamedTuple class dynamically
                    field_types = []
                    for name in field_names:
                        if name in Sig.output_fields:
                            field_types.append((name, Sig.output_fields[name].annotation))
                        else:
                            field_types.append((name, type(post[name])))
                    
                    # Create the NamedTuple class
                    DynamicStats = namedtuple('Stats', field_names)
                    
                    # Return the NamedTuple instance
                    return DynamicStats(*[post[name] for name in field_names])
            
            if typing.get_origin(ret_ann) is tuple:
                return tuple(post[n] for n in Sig.output_fields if n in post)
            if len(post) == 1:
                return next(iter(post.values()))
            return Example(**post)

        # pipe version keeps an Example so DSPy chains stay intact -----------
        def __ror__(self, lhs):
            if isinstance(lhs, tuple):
                lhs = dict(zip(sig_py.parameters, lhs))
            if not isinstance(lhs, dict):
                raise TypeError("lhs must be tuple or dict")
            ex = Example(**lhs); ex._signature = Sig
            return ex

        def __repr__(self):
            return f"<funky {fn.__name__}>"

    _Prog.module = default_mod  # expose raw DSPy module for optimizers
    _Prog._dspy  = default_mod  # synonym (shorter)
    return _Prog()

# pipeable wrappers around every DSPy module ----------------------------------

def _pipe_mod(ModCls: type[dspy.Module]):
    class W:
        def __init__(self):
            self._mods: dict[type[Signature], dspy.Module] = {}

        def __call__(self, *a, **k):
            return ModCls(*a, **k)

        def __ror__(self, ex: Example):
            sig = getattr(ex, "_signature", None)
            if sig is None:
                raise ValueError("missing _signature on lhs")
            mod = self._mods.get(sig) or ModCls(sig)
            self._mods[sig] = mod
            kw = {k: _to_text(v) for k, v in dict(ex).items()}
            res = mod(**kw)
            inputs = set(sig.input_fields)
            keep = {k: v for k, v in dict(res).items() if k not in inputs}
            typed = {k: _from_text(v, sig.output_fields[k].annotation) if k in sig.output_fields else v
                     for k, v in keep.items()}
            return Example(**typed)

        def __repr__(self):
            return f"<pipeable {ModCls.__name__}>"
    return W()

_mod = sys.modules[__name__]
for _n, _obj in vars(dspy).items():
    if isinstance(_obj, type) and issubclass(_obj, dspy.Module):
        setattr(_mod, _n.lower(), _pipe_mod(_obj))
if hasattr(_mod, "chainofthought"):
    setattr(_mod, "cot", getattr(_mod, "chainofthought"))

# -----------------------------------------------------------------------------
# decorator aliases mirroring real DSPy modules
# -----------------------------------------------------------------------------

def Predict(fn=None):
    return funky(fn, ModCls=dspy.Predict) if fn else lambda f: funky(f, ModCls=dspy.Predict)

def ChainOfThought(fn=None):
    return funky(fn, ModCls=dspy.ChainOfThought) if fn else lambda f: funky(f, ModCls=dspy.ChainOfThought)

def ReAct(fn=None):
    return funky(fn, ModCls=dspy.ReAct) if fn else lambda f: funky(f, ModCls=dspy.ReAct)

for _name in ("Predict", "ChainOfThought", "ReAct"):
    setattr(_mod, _name, globals()[_name])

# -----------------------------------------------------------------------------
# helper to register custom DSPy modules *or* wrap an existing instance
# -----------------------------------------------------------------------------

def register(cls, alias: str | None = None):
    """Expose a user-defined DSPy *class* as ``fd.<alias>`` and make it pipeable."""
    setattr(_mod, alias or cls.__name__.lower(), _pipe_mod(cls))


def funnier(mod, *, alias: str | None = None):
    """Return a *pythonic* wrapper around a **DSPy *instance***.

    Example
    -------
    ```python
    optim   = optimiser.compile(analyse.module, train)
    analyse_opt = fd.funnier(optim)       # normal call → Stats
    ```
    """
    Sig = mod.signature

    def _call(*a, _prediction: bool = False, **k):
        ex = Example(**Signature.python_signature(Sig).bind_partial(*a, **k).arguments)
        kwargs = {kk: _to_text(vv) for kk, vv in dict(ex).items()}
        pred: dspy.Prediction = mod(**kwargs)
        if _prediction:
            return pred
        post = {kk: _from_text(vv, Sig.output_fields[kk].annotation) for kk, vv in dict(pred).items()}
        if len(post) == 1:
            return next(iter(post.values()))
        return post

    _call.module = mod
    _call._dspy  = mod
    return _call

# expose helper in module namespace
setattr(_mod, "funnier", funnier)

def _extract_return_variable_names(fn) -> list[str]:
    """Extract variable names from return statements like 'return mean, above' or 'return answer'."""
    try:
        source = inspect.getsource(fn)
        tree = ast.parse(textwrap.dedent(source))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Tuple):
                    # Found a return statement with tuple: return mean, above
                    names = []
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Name):
                            names.append(elt.id)
                        else:
                            names.append(f"field{len(names)}")  # fallback for complex expressions
                    return names
                elif isinstance(node.value, ast.Name):
                    # Found a single variable return: return answer
                    return [node.value.id]
                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                    # Found return ClassName(...) pattern - look for NamedTuple fields
                    class_name = node.value.func.id
                    
                    # First, try to extract from keyword arguments: Stats(mean=mean, above=above)
                    if node.value.keywords:
                        return [kw.arg for kw in node.value.keywords if kw.arg]
                    
                    # If no keywords, try to find the class definition and extract field names
                    for class_node in ast.walk(tree):
                        if isinstance(class_node, ast.ClassDef) and class_node.name == class_name:
                            # Extract field names from annotations
                            field_names = []
                            for item in class_node.body:
                                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                    field_names.append(item.target.id)
                            if field_names:
                                return field_names
        return []
    except (OSError, TypeError, AttributeError, SyntaxError):
        # Handle cases where source code cannot be retrieved or parsed (e.g., interactive shell)
        # Try to extract variable names from the function's code object as a fallback
        try:
            code = fn.__code__
            
            # First, try to extract NamedTuple field names from code object constants
            # This handles the interactive mode case where NamedTuple is defined inside the function
            for const in code.co_consts:
                if hasattr(const, 'co_name') and hasattr(const, 'co_consts'):
                    # This is a code object, possibly for an inner class
                    # Check if it looks like a NamedTuple class
                    if (hasattr(const, 'co_names') and 
                        any('__annotations__' in name for name in const.co_names if isinstance(name, str))):
                        # Look for field names in the class constants
                        field_names = []
                        for class_const in const.co_consts:
                            if (isinstance(class_const, str) and 
                                class_const.isidentifier() and 
                                not class_const.startswith('_') and
                                class_const not in {'None', 'True', 'False'}):
                                # Skip the class name itself
                                if not (class_const.endswith('.Stats') or class_const == 'Stats'):
                                    field_names.append(class_const)
                        
                        # Filter out common non-field names
                        field_names = [name for name in field_names if name not in {
                            'float', 'int', 'str', 'list', 'dict', 'tuple', 'bool', 'type'
                        }]
                        
                        if field_names:
                            return field_names
            
            # For NamedTuple cases with keyword arguments like Stats(mean=mean, above=above),
            # the variable names often appear in co_names
            if code.co_names:
                # Filter out built-in names and common function names
                excluded = {'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'set', 'bool', 'type', 'object', 'NamedTuple'}
                candidates = [name for name in code.co_names if name not in excluded and not name.startswith('_')]
                
                # For single return values, return the first candidate
                if len(candidates) == 1:
                    return candidates
                
                # For multiple candidates, try to filter out class names and keep variable names
                # Look for patterns where we have both a class name and field names
                if len(candidates) > 1:
                    # If we have exactly 2 candidates and one looks like a class name (capitalized),
                    # return the others as field names
                    class_like = [name for name in candidates if name[0].isupper()]
                    var_like = [name for name in candidates if name[0].islower()]
                    if class_like and var_like:
                        return var_like
                    # Otherwise return all candidates (best effort)
                    return candidates
            
            # For tuple returns like "return result, count", check co_varnames
            if code.co_varnames:
                # co_varnames contains local variables, which includes return variables
                # Filter out function parameters by getting the parameter names from the signature
                try:
                    sig = inspect.signature(fn)
                    param_names = set(sig.parameters.keys())
                except:
                    param_names = set()
                
                # Filter out common parameter names, built-ins, and actual function parameters
                excluded = {'self', 'cls', 'args', 'kwargs', 'fn', 'func', 'function'} | param_names
                candidates = [name for name in code.co_varnames if name not in excluded and not name.startswith('_')]
                if candidates:
                    return candidates
                    
        except (AttributeError, TypeError):
            pass
        return []

def _extract_variable_descriptions(fn) -> dict[str, str]:
    """Extract descriptions from variable assignments like 'mean = "The average"'."""
    try:
        source = inspect.getsource(fn)
        tree = ast.parse(textwrap.dedent(source))
        descriptions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for assignments like: mean = "The average"
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        descriptions[var_name] = node.value.value
        return descriptions
    except (OSError, TypeError, AttributeError, SyntaxError):
        # Handle cases where source code cannot be retrieved or parsed
        return {}

# Export main functions and decorators
__all__ = [
    "funky",
    "Predict", 
    "ChainOfThought",
    "ReAct",
    "register",
    "funnier",
    "__version__",
]
