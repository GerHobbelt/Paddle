/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/sot/guards.h"

#if SOT_IS_SUPPORTED

#include <Python.h>
#include <frameobject.h>

#if !defined(PyObject_CallOneArg) && PY_VERSION_HEX < PY_3_9_0_HEX
static inline PyObject* PyObject_CallOneArg(PyObject* func, PyObject* arg) {
  return PyObject_CallFunctionObjArgs(func, arg, NULL);
}
#endif

bool LambdaGuard::check(PyObject* value) {
  PyObject* x = PyObject_CallOneArg(_guard_check_fn, value);
  if (x == nullptr) {
    PyErr_Clear();
    return false;
  }
  bool ret = PyObject_IsTrue(x);
  Py_DECREF(x);
  return ret;
}

bool GuardGroup::check(PyObject* value) {
  for (auto& guard : _guards) {
    if (!guard->check(value)) {
      return false;
    }
  }
  return true;
}

bool TypeMatchGuard::check(PyObject* value) {
  return Py_TYPE(value) == _expected;
}

bool ValueMatchGuard::check(PyObject* value) {
  if (value == _expected_value) {
    return true;
  }
  if (Py_TYPE(value) != _expected_type) {
    return false;
  }
  int result = PyObject_RichCompareBool(value, _expected_value, Py_EQ);
  // Check for exception
  if (result == -1) {
    PyErr_Clear();
    return false;
  }
  return result;
}

bool LengthMatchGuard::check(PyObject* value) {
  return PySequence_Size(value) == _expected;
}

bool DtypeMatchGuard::check(PyObject* value) {
  if (!paddle::pybind::PyCheckTensor(value)) {
    // TODO(zrr1999): PyCheckTensor only check if the object is a p_tensor_type.
    return false;
  }
  auto dtype =
      reinterpret_cast<paddle::pybind::TensorObject*>(value)->tensor.type();
  return phi::TransToProtoVarType(dtype) == _expected;
}

bool LayerMatchGuard::check(PyObject* value) {
  if (value != _layer_ptr) {
    return false;
  }
  PyObject* training = PyObject_GetAttrString(value, "training");
  return (training == Py_True) == _training;
}

#endif
