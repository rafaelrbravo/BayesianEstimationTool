"""Check manual tutorial code against the runnable BET examples.

Run from any working directory: python tests/check_manual_examples.py
No JAX/NumPyro import or inference run is required.
"""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANUAL=ROOT/"docs/Manual.md"
EXAMPLES=ROOT/"Examples"


def section(text,start,end):
    return text.split(start,1)[1].split(end,1)[0]


def blocks(text):
    result=[]
    for chunk in text.split("```python")[1:]:
        result.append(chunk.split("```",1)[0].strip())
    return result


def definitions(source):
    tree=ast.parse(source)
    return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def normalized(node):
    return ast.dump(node,include_attributes=False)


def calls(source,method):
    tree=ast.parse(source)
    result=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr==method:
            result.append(node)
    return result


def test_manual_examples():
    manual=MANUAL.read_text(encoding="utf-8")
    simple=section(manual,"## Part 2:","## Part 3:")
    correlated=section(manual,"## Part 3:","## Part 4:")
    saved=section(manual,"## Part 6:","## Part 7:")
    simple_blocks=blocks(simple)
    corr_blocks=blocks(correlated)
    save_blocks=blocks(saved)
    simple_code="\n\n".join(simple_blocks)
    corr_code="\n\n".join(corr_blocks)
    simple_defs=definitions(simple_code)
    corr_defs=definitions(corr_code)
    nuts=(EXAMPLES/"1_SimpleExampleNuts.py").read_text(encoding="utf-8")
    svi=(EXAMPLES/"2_SimpleExampleSvi.py").read_text(encoding="utf-8")
    save_nuts=(EXAMPLES/"3_SaveExampleNuts.py").read_text(encoding="utf-8")
    load_nuts=(EXAMPLES/"4_LoadExampleNuts.py").read_text(encoding="utf-8")
    save_svi=(EXAMPLES/"5_SaveExampleSvi.py").read_text(encoding="utf-8")
    load_svi=(EXAMPLES/"6_LoadExampleSvi.py").read_text(encoding="utf-8")
    for name in ("ModelFn","GlobalPriorFn","TrainLikelihoodFn","TestLikelihoodFn","ScoreFn"):
        assert name in simple_defs,f"Missing Part 2 definition: {name}"
        for label,source in (("simple NUTS",nuts),("simple SVI",svi)):
            assert normalized(simple_defs[name])==normalized(definitions(source)[name]),f"Part 2 {name} differs from {label}"
    for name in ("ModelFn","GlobalPriorFn"):
        assert name in corr_defs,f"Missing Part 3 definition: {name}"
        for label,source in (("save NUTS",save_nuts),("load NUTS",load_nuts),("save SVI",save_svi),("load SVI",load_svi)):
            assert normalized(corr_defs[name])==normalized(definitions(source)[name]),f"Part 3 {name} differs from {label}"

    # Check complete data generation, excluding comments and formatting.
    def before_model(source):
        return source.split("def ModelFn(",1)[0]
    data_names=("keyG","keyNoise","nPatients","nTimes","times","trueG","v0","trueTumorSize","tumorSize","data")
    def assignments(source):
        result={}
        for node in ast.parse(source).body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name):
                        result[target.id]=normalized(node.value)
                    elif isinstance(target,ast.Tuple):
                        for name in target.elts:
                            if isinstance(name,ast.Name):
                                result[name.id]=normalized(node.value)
        return result
    documented=assignments(simple_code)
    for label,source in (("simple NUTS",nuts),("simple SVI",svi)):
        example=assignments(before_model(source))
        for name in data_names:
            assert documented.get(name)==example.get(name),f"Part 2 data assignment {name} differs from {label}"

    # Match the documented SVI Train/Test calls to the executable SVI script.
    def compare_calls(documented_source,example_source,method,expected):
        doc_calls=calls(documented_source,method)
        example_calls=calls(example_source,method)
        assert len(doc_calls)==expected,f"Expected {expected} documented {method} calls, got {len(doc_calls)}"
        assert len(example_calls)>=expected
        for i,(a,b) in enumerate(zip(doc_calls,example_calls)):
            assert normalized(a)==normalized(b),f"Documented {method} call {i+1} differs from SVI example"
    compare_calls(simple_code,svi,"Train",1)
    compare_calls(simple_code,svi,"Test",2)

    # The manual's Save/Load commands must be the actual paired filenames/calls.
    save_code="\n\n".join(save_blocks)
    assert 'est.Save("SimpleTumorGrowthNuts.npz")' in save_code
    assert 'est.Save("SimpleTumorGrowthNuts.npz")' in save_nuts
    assert 'est=bet.Estimator.Load("SimpleTumorGrowthNuts.npz",ModelFn=ModelFn,GlobalPriorFn=GlobalPriorFn,data=data)' in save_code
    assert 'est=bet.Estimator.Load("SimpleTumorGrowthNuts.npz",ModelFn=ModelFn,GlobalPriorFn=GlobalPriorFn,data=data)' in load_nuts
    assert 'est.Save("SimpleTumorGrowthSvi.npz")' in save_svi
    assert 'bet.Estimator.Load("SimpleTumorGrowthSvi.npz"' in load_svi
    print("BET manual/example consistency checks passed.")


if __name__=="__main__":
    test_manual_examples()
