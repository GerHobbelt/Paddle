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
#pragma once

#include <Python.h>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/core/framework/heter_service.pb.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/pybind.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
#define PYBIND11_DETAILED_ERROR_MESSAGES
#if SOT_IS_SUPPORTED

class GuardBase {
 public:
  GuardBase() = default;

  bool check_pybind(py::handle value) { return check(value.ptr()); }

  virtual bool check(PyObject* value) = 0;
  virtual ~GuardBase() = default;
};

class LambdaGuard : public GuardBase {
 public:
  explicit LambdaGuard(PyObject* guard_check_fn)
      : _guard_check_fn(guard_check_fn) {}

  explicit LambdaGuard(const py::function& guard_check_fn)
      : _guard_check_fn(guard_check_fn.ptr()) {
    Py_INCREF(_guard_check_fn);
  }

  ~LambdaGuard() { Py_DECREF(_guard_check_fn); }

  bool check(PyObject* value);

 private:
  PyObject* _guard_check_fn;
};

class GuardGroup : public GuardBase {
 public:
  explicit GuardGroup(std::vector<std::shared_ptr<GuardBase>> guards) {
    for (auto& guard : guards) {
      if (auto group = dynamic_cast<GuardGroup*>(guard.get())) {
        _guards.insert(
            _guards.end(), group->_guards.begin(), group->_guards.end());
      } else {
        _guards.push_back(std::move(guard));
      }
    }
  }
  bool check(PyObject* value);

 private:
  std::vector<std::shared_ptr<GuardBase>> _guards;
};

class TypeMatchGuard : public GuardBase {
 public:
  explicit TypeMatchGuard(PyObject* type_ptr)
      : _expected(reinterpret_cast<PyTypeObject*>(type_ptr)) {}

  explicit TypeMatchGuard(const py::type& py_type)
      : _expected(reinterpret_cast<PyTypeObject*>(py_type.ptr())) {}

  bool check(PyObject* value);

 private:
  PyTypeObject* _expected;
};

class ValueMatchGuard : public GuardBase {
 public:
  explicit ValueMatchGuard(PyObject* value_ptr)
      : _expected_value(value_ptr), _expected_type(value_ptr->ob_type) {}

  explicit ValueMatchGuard(const py::object& py_value)
      : _expected_value(py_value.ptr()),
        _expected_type(Py_TYPE(py_value.ptr())) {
    Py_INCREF(_expected_value);
  }

  ~ValueMatchGuard() { Py_DECREF(_expected_value); }

  bool check(PyObject* value);

 private:
  PyObject* _expected_value;
  PyTypeObject* _expected_type;
};

class LengthMatchGuard : public GuardBase {
 public:
  explicit LengthMatchGuard(Py_ssize_t length) : _expected(length) {}

  bool check(PyObject* value);

 private:
  Py_ssize_t _expected;
};

class DtypeMatchGuard : public GuardBase {
 public:
  explicit DtypeMatchGuard(const paddle::framework::proto::VarType& dtype_ptr)
      : _expected(dtype_ptr.type()) {}

  explicit DtypeMatchGuard(const phi::DataType& dtype_ptr)
      : _expected(phi::TransToProtoVarType(dtype_ptr)) {}

  bool check(PyObject* value);

 private:
  int _expected;
};

class LayerMatchGuard : public GuardBase {
 public:
  explicit LayerMatchGuard(PyObject* layer_ptr) : _layer_ptr(layer_ptr) {
    _training = PyObject_GetAttrString(layer_ptr, "training") == Py_True;
  }

  explicit LayerMatchGuard(const py::object& layer_obj)
      : _layer_ptr(layer_obj.ptr()), _training(layer_obj.attr("training")) {}

  bool check(PyObject* value);

 private:
  PyObject* _layer_ptr;
  bool _training;
};

#endif
