"""Exercise the actual numeric helpers without requiring Pi audio/TFLite."""
import ast
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def function(path,name,scope):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),scope)
    return scope[name]

class QuantizationTests(unittest.TestCase):
    def test_scalar_saturates_and_rounds(self):
        quantize=function(ROOT/"tinyml-waveforms/pi.py","quantize_scalar",{"np":np})
        self.assertEqual(quantize(100,0.01,0,np.int8).item(),127)
        self.assertEqual(quantize(-100,0.01,0,np.int8).item(),-128)
        self.assertEqual(quantize(0.016,0.01,0,np.int8).item(),2)
        with self.assertRaises(ValueError):quantize(1,0,0,np.int8)

    def test_mfcc_quantization_cannot_wrap(self):
        preprocess=function(ROOT/"wake-word/wakeWordSimpleEval.py","preprocess",{"np":np,"SAMPLE_RATE":16000,"NUM_MFCC":13,"mfcc":lambda *a,**kw:np.array([[100.0,-100.0]])})
        value=preprocess(np.ones(10),[{"quantization":(0.01,0),"dtype":np.int8}])
        self.assertEqual(value.ravel().tolist(),[127,-128])

if __name__=="__main__":unittest.main()
