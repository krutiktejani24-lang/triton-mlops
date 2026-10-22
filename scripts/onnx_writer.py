"""
Minimal, dependency-free ONNX writer.

It serialises an ONNX ModelProto by hand (protobuf wire format), so models can
be exported without installing the `onnx` / `skl2onnx` packages. It supports
exactly what this project needs: a feed-forward network made of
Flatten -> (Gemm -> Relu)* -> Gemm -> Softmax.

The produced file is a normal ONNX model (IR v8, opset 13) that loads in
ONNX Runtime and in Triton's `onnxruntime_onnx` backend.
"""
import struct
import numpy as np

# ---------------------------------------------------------------- protobuf --

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _str(field: int, s: str) -> bytes:
    return _len(field, s.encode("utf-8"))


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


# ------------------------------------------------------------ ONNX objects --

FLOAT = 1  # TensorProto.DataType.FLOAT


def _tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    out = b"".join(_int(1, d) for d in arr.shape)       # dims
    out += _int(2, FLOAT)                                # data_type
    out += _str(8, name)                                 # name
    out += _len(9, arr.tobytes())                        # raw_data
    return out


def _attr_int(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT = 2)
    return _str(1, name) + _int(3, value) + _int(20, 2)


def _node(op: str, inputs, outputs, name: str, attrs=()) -> bytes:
    out = b"".join(_str(1, i) for i in inputs)
    out += b"".join(_str(2, o) for o in outputs)
    out += _str(3, name) + _str(4, op)
    out += b"".join(_len(5, a) for a in attrs)
    return out


def _value_info(name: str, dims) -> bytes:
    shape = b""
    for d in dims:
        shape += _len(1, _str(2, d) if isinstance(d, str) else _int(1, d))
    tensor_type = _int(1, FLOAT) + _len(2, shape)
    return _str(1, name) + _len(2, _len(1, tensor_type))


# ------------------------------------------------------------------- model --

def build_mlp_onnx(layers, input_name, output_name, input_dims,
                   flatten=False, graph_name="mlp") -> bytes:
    """
    layers     : list of (W, b); W has shape (in, out). ReLU between layers,
                 Softmax after the last one.
    input_dims : e.g. ["batch", 8] or ["batch", 1, 8, 8]
    flatten    : insert a Flatten node first (for image tensors).
    """
    nodes, inits = [], []
    current = input_name

    if flatten:
        nodes.append(_node("Flatten", [current], ["flat"], "flatten",
                           [_attr_int("axis", 1)]))
        current = "flat"

    for i, (W, b) in enumerate(layers):
        inits.append(_tensor(f"W{i}", W))
        inits.append(_tensor(f"b{i}", b))
        out = f"fc{i}"
        nodes.append(_node("Gemm", [current, f"W{i}", f"b{i}"], [out], f"gemm{i}"))
        current = out
        if i < len(layers) - 1:
            nodes.append(_node("Relu", [current], [f"relu{i}"], f"relu{i}"))
            current = f"relu{i}"

    nodes.append(_node("Softmax", [current], [output_name], "softmax",
                       [_attr_int("axis", 1)]))

    n_out = layers[-1][0].shape[1]
    graph = b"".join(_len(1, n) for n in nodes)
    graph += _str(2, graph_name)
    graph += b"".join(_len(5, t) for t in inits)
    graph += _len(11, _value_info(input_name, input_dims))
    graph += _len(12, _value_info(output_name, ["batch", n_out]))

    model = _int(1, 8)                         # ir_version
    model += _str(2, "triton-mlops-project")   # producer_name
    model += _len(7, graph)                    # graph
    model += _len(8, _int(2, 13))              # opset_import (default domain, v13)
    return model
