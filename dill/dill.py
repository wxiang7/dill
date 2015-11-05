# -*- coding: utf-8 -*-
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/pathos/browser/dill/LICENSE
"""
dill: a utility for serialization of python objects

Based on code written by Oren Tirosh and Armin Ronacher.
Extended to a (near) full set of the builtin types (in types module),
and coded to the pickle interface, by <mmckerns@caltech.edu>.
Initial port to python3 by Jonathan Dobson, continued by mmckerns.
Test against "all" python types (Std. Lib. CH 1-15 @ 2.7) by mmckerns.
Test against CH16+ Std. Lib. ... TBD.
"""

import logging
import os
import sys
import marshal
import gc

from weakref import ReferenceType, ProxyType, CallableProxyType
from functools import partial
from operator import itemgetter, attrgetter
from pickle import HIGHEST_PROTOCOL, PicklingError, UnpicklingError
from os.path import abspath, commonprefix, join
from six import PY3, iteritems, get_function_globals, get_function_code, get_function_defaults, get_function_closure
from six import get_method_function, get_method_self

import __main__ as _main_module


if PY3:
    import builtins as __builtin__
    from pickle import _Pickler as StockPickler, _Unpickler as StockUnpickler

    BufferType = memoryview                   # XXX: unregistered
    ClassType = type                          # no 'old-style' classes
    EllipsisType = type(Ellipsis)
    NotImplementedType = type(NotImplemented)
    SliceType = slice
    TypeType = type                           # 'new-style' classes #XXX: unregistered
    XRangeType = range
    DictProxyType = type(object.__dict__)
else:
    import __builtin__
    from pickle import Pickler as StockPickler, Unpickler as StockUnpickler
    from types import CodeType, FunctionType, ClassType, MethodType, \
        XRangeType, SliceType, NotImplementedType, EllipsisType, ModuleType, \
        BuiltinMethodType, TypeType

try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL

# new in python2.5
if sys.hexversion >= 0x20500f0:
    from types import MemberDescriptorType, GetSetDescriptorType
# new in python3.3
if sys.hexversion < 0x03030000:
    FileNotFoundError = IOError

try:
    import ctypes

    HAS_CTYPES = True
    IS_PYPY = True
except ImportError:
    HAS_CTYPES = False
    IS_PYPY = False

try:
    from numpy import ufunc as NumpyUfuncType
    from numpy import ndarray as NumpyArrayType

    def ndarraysubclassinstance(obj):
        try:  # check if is ndarray, and elif is subclass of ndarray
            if getattr(obj, '__class__', type) is NumpyArrayType:
                return False
            elif not isinstance(obj, NumpyArrayType):
                return False
        except ReferenceError:
            return False  # handle 'R3' weakref in 3.x
        # verify that __reduce__ has not been overridden
        NumpyInstance = NumpyArrayType((0,), 'int8')
        if id(obj.__reduce_ex__) == id(NumpyInstance.__reduce_ex__) and \
                        id(obj.__reduce__) == id(NumpyInstance.__reduce__): return True
        return False
except ImportError:
    NumpyUfuncType = None
    NumpyArrayType = None

    def ndarraysubclassinstance(obj):
        return False

# make sure to add these 'hand-built' types to _typemap
if PY3:
    CellType = type((lambda x: lambda y: x)(0).__closure__[0])
else:
    CellType = type((lambda x: lambda y: x)(0).func_closure[0])

WrapperDescriptorType = type(type.__repr__)
MethodDescriptorType = type(type.__dict__['mro'])
MethodWrapperType = type([].__repr__)
PartialType = type(partial(int, base=2))
SuperType = type(super(Exception, TypeError()))
ItemGetterType = type(itemgetter(0))
AttrGetterType = type(attrgetter('__repr__'))
FileType = type(open(os.devnull, 'rb', buffering=0))
TextWrapperType = type(open(os.devnull, 'r', buffering=-1))
BufferedRandomType = type(open(os.devnull, 'r+b', buffering=-1))
BufferedReaderType = type(open(os.devnull, 'rb', buffering=-1))
BufferedWriterType = type(open(os.devnull, 'wb', buffering=-1))

try:
    from _pyio import open as _open

    PyTextWrapperType = type(_open(os.devnull, 'r', buffering=-1))
    PyBufferedRandomType = type(_open(os.devnull, 'r+b', buffering=-1))
    PyBufferedReaderType = type(_open(os.devnull, 'rb', buffering=-1))
    PyBufferedWriterType = type(_open(os.devnull, 'wb', buffering=-1))
except ImportError:
    PyTextWrapperType = PyBufferedRandomType = PyBufferedReaderType = PyBufferedWriterType = None

try:
    from cStringIO import StringIO, InputType, OutputType
except ImportError:
    if PY3:
        from io import BytesIO as StringIO
    else:
        from StringIO import StringIO
    InputType = OutputType = None

try:
    __IPYTHON__ is True        # is ipython
    ExitType = None            # IPython.core.autocall.ExitAutocall
    singletontypes = ['exit', 'quit', 'get_ipython']
except NameError:
    try:
        ExitType = type(exit)  # apparently 'exit' can be removed
    except NameError:
        ExitType = None
    singletontypes = []

log = logging.getLogger("dill")


def _proxy_helper(obj):  # a dead proxy returns a reference to None
    """get memory address of proxy's reference object"""
    try:  # FIXME: has to be a smarter way to identify if it's a proxy
        address = int(repr(obj).rstrip('>').split(' at ')[-1], base=16)
    except ValueError:  # has a repr... is thus probably not a proxy
        address = id(obj)
    return address


def _locate_object(address, module=None):
    """get object located at the given memory address (inverse of id(obj))"""
    special = [None, True, False]
    for obj in special:
        if address == id(obj): return obj
    if module:
        if PY3:
            objects = iter(module.__dict__.values())
        else:
            objects = module.__dict__.itervalues()
    else:
        objects = iter(gc.get_objects())
    for obj in objects:
        if address == id(obj):
            return obj
    # all bad below... nothing found so throw ReferenceError or TypeError
    from weakref import ReferenceError
    try:
        address = hex(address)
    except TypeError:
        raise TypeError("'%s' is not a valid memory address" % str(address))
    raise ReferenceError("Cannot reference object at '%s'" % address)


def _trace(boolean):
    """print a trace through the stack when pickling; useful for debugging"""
    if boolean:
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARN)


def copy(obj, *args, **kwds):
    """use pickling to 'copy' an object"""
    return loads(dumps(obj, *args, **kwds))


def dump(obj, file_out, protocol=None, byref=None, fmode=None, recurse=None, **kwargs):
    """pickle an object to a file"""
    from .settings import settings
    strictio = False  # FIXME: strict=True needs cleanup
    protocol = protocol or settings['protocol']
    byref = byref or settings['byref']
    fmode = fmode or settings['fmode']
    recurse = recurse or settings['recurse']
    pik = Pickler(file_out, protocol)
    pik._main = _main_module
    # apply kwd settings
    pik._byref = bool(byref)
    pik._strictio = bool(strictio)
    pik._fmode = fmode
    pik._recurse = bool(recurse)
    pik.ship_path = []
    for k, v in iteritems(kwargs):
        if k == 'ship_path':
            v = [abspath(p) for p in v]
        pik.__dict__[k] = v
    # hack to catch subclassed numpy array instances
    if NumpyArrayType and ndarraysubclassinstance(obj):
        @register(type(obj))
        def save_numpy_array(pickler, obj):
            log.info("Nu: (%s, %s)" % (obj.shape, obj.dtype))
            npdict = getattr(obj, '__dict__', None)
            f, args, state = obj.__reduce__()
            pik.save_reduce(_create_array, (f, args, state, npdict), obj=obj)
            log.info("# Nu")
            return
    # end hack
    pik.dump(obj)
    return


def dumps(obj, protocol=None, byref=None, fmode=None, recurse=None, **kwargs):
    """pickle an object to a string"""
    file_out = StringIO()
    dump(obj, file_out, protocol, byref, fmode, recurse, **kwargs)
    pickled = file_out.getvalue()

    if 'compress' in kwargs and kwargs['compress']:
        from zlib import compress
        pickled = compress(pickled)
    return pickled


def load(file_in, **kwargs):
    """unpickle an object from a file"""
    pik = Unpickler(file_in)
    pik._main = _main_module
    for k, v in iteritems(kwargs):
        pik.__dict__[k] = v
    obj = pik.load()
    if type(obj).__module__ == _main_module.__name__:  # point obj class to main
        try:
            obj.__class__ == getattr(_main_module, type(obj).__name__)
        except AttributeError:
            pass  # defined in a file
    return obj


def loads(str_pickled, **kwargs):
    """unpickle an object from a string"""
    if 'compress' in kwargs and kwargs['compress']:
        from zlib import decompress
        str_pickled = decompress(str_pickled)
    file_in = StringIO(str_pickled)
    return load(file_in, **kwargs)


# Pickle the Interpreter Session
def _module_map():
    """get map of imported modules"""
    from collections import defaultdict
    mod_map = defaultdict(list)
    for mod_name, mod in iteritems(sys.modules):
        if mod is not None:
            for obj_name, obj in iteritems(mod.__dict__):
                mod_map[obj_name].append((obj, mod_name))
    return mod_map


def _lookup_module(mod_map, name, obj, main_module):
    """lookup name if module is imported"""
    for mod_obj, mod_name in mod_map[name]:
        if mod_obj is obj and mod_name != main_module.__name__:
            return mod_name


def _stash_modules(main_module):
    mod_map = _module_map()
    imported = []
    original = {}
    for name, obj in iteritems(main_module.__dict__):
        source_module = _lookup_module(mod_map, name, obj, main_module)
        if source_module:
            imported.append((source_module, name))
        else:
            original[name] = obj
    if len(imported):
        import types
        new_mod = types.ModuleType(main_module.__name__)
        new_mod.__dict__.update(original)
        new_mod.__dill_imported = imported
        return new_mod
    else:
        return original


def _restore_modules(main_module):
    if '__dill_imported' not in main_module.__dict__:
        return
    imports = main_module.__dict__.pop('__dill_imported')
    for module, name in imports:
        exec ("from %s import %s" % (module, name), main_module.__dict__)


class MetaCatchingDict(dict):

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __missing__(self, key):
        if issubclass(key, type):
            return save_type
        else:
            raise KeyError()


class ProxyModule(dict):

    def __init__(self, mod_name):
        self.mod_name = mod_name

    def __getattr__(self, item):
        mod = sys.modules[self.mod_name]
        return getattr(mod, item)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def proxy(self, item):

        def invoke(*args, **kwargs):
            self.__getattr__(item)(*args, **kwargs)

        return invoke


FUNC_GLOBAL = '+'
CLASS_DICT = '='
from pickle import SETITEM, SETITEMS


# Extend the Pickler
class Pickler(StockPickler):

    """python's Pickler extended to interpreter sessions"""
    from .settings import settings
    dispatch = MetaCatchingDict(StockPickler.dispatch.copy())
    _main = None
    _byref = settings['byref']
    _strictio = False
    _fmode = settings['fmode']
    _recurse = settings['recurse']

    def __init__(self, *args, **kwds):
        _byref = kwds.pop('byref', Pickler._byref)
        _fmode = kwds.pop('fmode', Pickler._fmode)
        _recurse = kwds.pop('recurse', Pickler._recurse)
        StockPickler.__init__(self, *args, **kwds)
        self._main = _main_module
        self._byref = _byref
        self._fmode = _fmode
        self._recurse = _recurse

    def save_reduce(self, func, args, state=None,
                    listitems=None, dictitems=None, obj=None, func_globals=None, class_dict=None):
        from pickle import TupleType, NEWOBJ, REDUCE, BUILD
        # This API is called by some subclasses

        # Assert that args is a tuple or None
        if not isinstance(args, TupleType):
            raise PicklingError("args from reduce() should be a tuple")

        # Assert that func is callable
        if not hasattr(func, '__call__'):
            raise PicklingError("func from reduce should be callable")

        save = self.save
        write = self.write

        # Protocol 2 special case: if func's name is __newobj__, use NEWOBJ
        if self.proto >= 2 and getattr(func, "__name__", "") == "__newobj__":
            # A __reduce__ implementation can direct protocol 2 to
            # use the more efficient NEWOBJ opcode, while still
            # allowing protocol 0 and 1 to work normally.  For this to
            # work, the function returned by __reduce__ should be
            # called __newobj__, and its first argument should be a
            # new-style class.  The implementation for __newobj__
            # should be as follows, although pickle has no way to
            # verify this:
            #
            # def __newobj__(cls, *args):
            #     return cls.__new__(cls, *args)
            #
            # Protocols 0 and 1 will pickle a reference to __newobj__,
            # while protocol 2 (and above) will pickle a reference to
            # cls, the remaining args tuple, and the NEWOBJ code,
            # which calls cls.__new__(cls, *args) at unpickling time
            # (see load_newobj below).  If __reduce__ returns a
            # three-tuple, the state from the third tuple item will be
            # pickled regardless of the protocol, calling __setstate__
            # at unpickling time (see load_build below).
            #
            # Note that no standard __newobj__ implementation exists;
            # you have to provide your own.  This is to enforce
            # compatibility with Python 2.2 (pickles written using
            # protocol 0 or 1 in Python 2.3 should be unpicklable by
            # Python 2.2).
            cls = args[0]
            if not hasattr(cls, "__new__"):
                raise PicklingError(
                    "args[0] from __newobj__ args has no __new__")
            if obj is not None and cls is not obj.__class__:
                raise PicklingError(
                    "args[0] from __newobj__ args has the wrong class")
            args = args[1:]
            save(cls)
            save(args)
            write(NEWOBJ)
        else:
            save(func)
            save(args)
            write(REDUCE)

        if obj is not None:
            self.memoize(obj)

        # More new special cases (that work with older protocols as
        # well): when __reduce__ returns a tuple with 4 or 5 items,
        # the 4th and 5th item should be iterators that provide list
        # items and dict items (as (key, value) tuples), or None.

        if func_globals is not None:
            self._batch_func_globals(func_globals)

        if listitems is not None:
            self._batch_appends(listitems)

        if dictitems is not None:
            self._batch_setitems(dictitems)

        if class_dict is not None:
            self._batch_class_dict(class_dict)

        if state is not None:
            save(state)
            write(BUILD)

    def _batch_class_dict(self, class_dict):
        self.save(class_dict)
        self.write(CLASS_DICT)

    def _batch_func_globals(self, func_globals):
        # Helper to batch up FUNC_GLOBAL_ITEM sequences; proto >= 1 only
        self.save(func_globals)
        self.write(FUNC_GLOBAL)

    def write_lines(self, lines):
        if PY3:
            self.write(bytes(lines, 'UTF-8'))
        else:
            self.write(lines)


class Unpickler(StockUnpickler):
    """python's Unpickler extended to interpreter sessions and more types"""
    _main = None

    def find_class(self, module, name):
        if (module, name) == ('__builtin__', '__main__'):
            return self._main.__dict__
        elif (module, name) == ('__builtin__', '__dict__'):
            return __builtin__.__dict__
        elif (module, name) == ('__builtin__', 'NoneType'):
            return type(None)
        try:
            clz = StockUnpickler.find_class(self, module, name)
        except (ImportError, AttributeError):
            log.info('creating module for "%s"."%s"' % (module, name))
            mod = ProxyModule(module)
            return mod.proxy(name)

        return clz

    def load_func_global_item(self):
        stack = self.stack
        globs = stack.pop()
        func = stack[-1]
        if globs:
            for k, v in iteritems(globs):
                func.func_globals[k] = v
        self._func_globals[id(globs)].append(func)
    StockUnpickler.dispatch[FUNC_GLOBAL] = load_func_global_item

    def load_class_dict(self):
        stack = self.stack
        class_dict = stack.pop()
        class_obj = stack[-1]
        for k, v in iteritems(class_dict):
            class_obj.__dict__[k] = v
    StockUnpickler.dispatch[CLASS_DICT] = load_class_dict

    def load_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        dict = stack[-1]
        dict[key] = value

        if is_dill(self) and id(dict) in self._func_globals:
            funcs = self._func_globals[id(dict)]
            for func in funcs:
                func.func_globals[key] = value
    StockUnpickler.dispatch[SETITEM] = load_setitem

    def load_setitems(self):
        stack = self.stack
        mark = self.marker()
        dict = stack[mark - 1]
        for i in range(mark + 1, len(stack), 2):
            dict[stack[i]] = stack[i + 1]

        if is_dill(self) and id(dict) in self._func_globals:
            funcs = self._func_globals[id(dict)]
            for i in range(mark + 1, len(stack), 2):
                for func in funcs:
                    func.func_globals[stack[i]] = stack[i + 1]

        del stack[mark:]
    StockUnpickler.dispatch[SETITEMS] = load_setitems

    def __init__(self, *args, **kwds):
        StockUnpickler.__init__(self, *args, **kwds)
        from collections import defaultdict
        self._main = _main_module
        self._func_globals = defaultdict(list)

pickle_dispatch_copy = StockPickler.dispatch.copy()


def pickle(t, func):
    """expose dispatch table for user-created extensions"""
    Pickler.dispatch[t] = func


def register(t):
    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy


def _create_type_map():
    import types
    if PY3:
        d = iteritems(dict(list(__builtin__.__dict__.items()) + list(types.__dict__.items())))
        builtin = 'builtins'
    else:
        d = iteritems(types.__dict__)
        builtin = '__builtin__'

    for key, value in d:
        if getattr(value, '__module__', None) == builtin and type(value) is type:
            yield key, value

_reverse_type_map = dict(_create_type_map())
_reverse_type_map.update({
    'CellType': CellType,
    'WrapperDescriptorType': WrapperDescriptorType,
    'MethodDescriptorType': MethodDescriptorType,
    'MethodWrapperType': MethodWrapperType,
    'PartialType': PartialType,
    'SuperType': SuperType,
    'ItemGetterType': ItemGetterType,
    'AttrGetterType': AttrGetterType,
    'FileType': FileType,
    'BufferedRandomType': BufferedRandomType,
    'BufferedReaderType': BufferedReaderType,
    'BufferedWriterType': BufferedWriterType,
    'TextWrapperType': TextWrapperType,
    'PyBufferedRandomType': PyBufferedRandomType,
    'PyBufferedReaderType': PyBufferedReaderType,
    'PyBufferedWriterType': PyBufferedWriterType,
    'PyTextWrapperType': PyTextWrapperType,
})
if ExitType:
    _reverse_type_map['ExitType'] = ExitType
if InputType:
    _reverse_type_map['InputType'] = InputType
    _reverse_type_map['OutputType'] = OutputType
_type_map = dict((v, k) for k, v in iteritems(_reverse_type_map))


def _unmarshal(string):
    return marshal.loads(string)


def _load_type(name):
    return _reverse_type_map[name]


def _create_type(type_obj, *args):
    return type_obj(*args)


def _create_function(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, mod_name=None):
    # same as FunctionType, but enable passing __dict__ to new function,
    # __dict__ is the storehouse for attributes added after function creation
    log.info('loading function: ' + fname)
    fdict = fdict or dict()
    fglobals = fglobals or {}
    func = FunctionType(fcode, fglobals, fname, fdefaults, fclosure)
    func.__dict__.update(fdict)
    func.__module__ = mod_name
    return func


def _create_ftype(ftype_obj, func, args, kwds):
    kwds = kwds or {}
    args = args or ()
    return ftype_obj(func, *args, **kwds)


class _itemgetter_helper(object):
    def __init__(self):
        self.items = []

    def __getitem__(self, item):
        self.items.append(item)
        return


class _attrgetter_helper(object):
    def __init__(self, attrs, index=None):
        self.attrs = attrs
        self.index = index

    def __getattribute__(self, attr):
        attrs = object.__getattribute__(self, "attrs")
        index = object.__getattribute__(self, "index")
        if index is None:
            index = len(attrs)
            attrs.append(attr)
        else:
            attrs[index] = ".".join([attrs[index], attr])
        return type(self)(attrs, index)


def _create_weakref(obj, *args):
    from weakref import ref
    if obj is None:  # it's dead
        from six.moves import UserDict
        return ref(UserDict(), *args)
    return ref(obj, *args)


def _create_weakproxy(obj, callable=False, *args):
    from weakref import proxy
    if obj is None:  # it's dead
        if callable:
            return proxy(lambda x: x, *args)
        from six.moves import UserDict
        return proxy(UserDict(), *args)
    return proxy(obj, *args)


def _eval_repr(repr_str):
    return eval(repr_str)


def _create_array(f, args, state, npdict=None):
    array = f(*args)
    array.__setstate__(state)
    if npdict is not None:
        array.__dict__.update(npdict)
    return array


def _getattr(objclass, name, repr_str):
    # hack to grab the reference directly
    try:
        attr = repr_str.split("'")[3]
        return eval(attr + '.__dict__["' + name + '"]')
    except:
        attr = getattr(objclass, name)
        if name == '__dict__':
            attr = attr[name]
        return attr


def _get_attr(self, name):
    return getattr(self, name, None) or getattr(__builtin__, name)


def _dict_from_dictproxy(dictproxy):
    _dict = dictproxy.copy()  # convert dictproxy to dict
    _dict.pop('__dict__', None)
    _dict.pop('__weakref__', None)
    return _dict


def _from_ship_path(path, ship_path):
    return any([commonprefix([pth, path]) == pth for pth in ship_path])


def _import_module(import_name, safe=False):
    try:
        if '.' in import_name:
            items = import_name.split('.')
            mod_name = '.'.join(items[:-1])
            obj_name = items[-1]
            try:
                obj = getattr(__import__(mod_name, None, None, [obj_name]), obj_name)
            except ImportError:
                # sub module
                mod = ModuleType(import_name)
                sys.modules[import_name] = mod
                obj = mod
            return obj
        else:
            try:
                return __import__(import_name)
            except ImportError:
                mod = ModuleType(import_name)
                sys.modules[import_name] = mod
                return mod
    except (ImportError, AttributeError):
        if safe:
            return None
        raise


def _importable(import_name, safe=False):
    try:
        if '.' in import_name:
            items = import_name.split('.')
            module = '.'.join(items[:-1])
            obj = items[-1]
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if safe:
            return None
        raise


def _locate_function(obj, pickler):
    if obj.__module__ in ['__main__', None]:
        return False
    full_name = obj.__module__ + '.' + obj.__name__

    if _importable(full_name, safe=True) is obj and is_dill(pickler):
        mod = sys.modules[obj.__module__]
        mod_file = mod.__file__
        in_ship_dir = is_dill(pickler) and _from_ship_path(mod_file, pickler.ship_path)
        return not in_ship_dir
    else:
        return False


@register(CodeType)
def save_code(pickler, obj):
    log.info("Co: %s" % obj)
    pickler.save_reduce(_unmarshal, (marshal.dumps(obj),), obj=obj)
    log.info("# Co")
    return


@register(FunctionType)
def save_function(pickler, obj):
    if not _locate_function(obj, pickler):
        log.info("F1: %s" % obj)
        globs = get_function_globals(obj)
        mod_name = obj.__module__

        pickler.save_reduce(_create_function, (get_function_code(obj),
                                               {},
                                               obj.__name__,
                                               get_function_defaults(obj),
                                               get_function_closure(obj),
                                               obj.__dict__,
                                               mod_name), obj=obj, func_globals=globs)

        log.info("# F1 %s" % obj)
    else:
        log.info("F2: %s" % obj)
        StockPickler.save_global(pickler, obj)
        log.info("# F2 %s" % obj)
    return


@register(dict)
def save_module_dict(pickler, obj):
    if is_dill(pickler) and obj == pickler._main.__dict__:
        log.info("D1: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        pickler.write_lines('c__builtin__\n__main__\n')
        log.info("# D1 <dict%s" % str(obj.__repr__).split('dict')[-1])
    elif is_dill(pickler) and obj == __builtin__.__dict__:
        log.info("D6: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        pickler.write('c__builtin__\n__dict__\n')
        log.info("# D6 <dict%s" % str(obj.__repr__).split('dict')[-1])
    elif not is_dill(pickler) and obj == _main_module.__dict__:
        log.info("D3: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        pickler.write('c__main__\n__dict__\n')
        log.info("# D3 <dict%s" % str(obj.__repr__).split('dict')[-1])
    elif '__name__' in obj and obj != _main_module.__dict__ \
            and obj is getattr(_importable(obj['__name__'], True), '__dict__', None) \
            and '__file__' in obj and _from_ship_path(obj['__file__'], pickler.ship_path):
        log.info("D5: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        StockPickler.save_dict(pickler, obj)
        log.info("# D5 <dict%s" % str(obj.__repr__).split('dict')[-1])
    elif '__name__' in obj and obj != _main_module.__dict__ \
            and obj is getattr(_importable(obj['__name__'], True), '__dict__', None):
        log.info("D4: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        pickler.write('c%s\n__dict__\n' % obj['__name__'])
        log.info("# D4 <dict%s" % str(obj.__repr__).split('dict')[-1])
    else:
        log.info("D2: <dict%s" % str(obj.__repr__).split('dict')[-1])  # obj
        StockPickler.save_dict(pickler, obj)
        log.info("# D2 <dict%s" % str(obj.__repr__).split('dict')[-1])
    return


@register(ClassType)
def save_classobj(pickler, obj):
    if obj.__module__ == '__main__':
        log.info("C1: %s" % obj)
        pickler.save_reduce(ClassType, (obj.__name__, obj.__bases__,
                                        obj.__dict__), obj=obj)
        log.info("# C1")
    elif obj.__module__ in sys.modules and _from_ship_path(sys.modules[obj.__module__].__file__, pickler.ship_path):
        log.info("C3: %s" % obj)
        class_dict = obj.__dict__
        class_funcs = dict([(k, v) for k, v in iteritems(class_dict) \
                            if isinstance(v, FunctionType) or isinstance(v, classmethod) or isinstance(v, staticmethod)])
        class_non_funcs = dict([(k, v) for k, v in iteritems(class_dict) \
                                if not (isinstance(v, FunctionType) or isinstance(v, classmethod) or isinstance(v, staticmethod))])
        pickler.save_reduce(ClassType, (obj.__name__, obj.__bases__,
                                        class_non_funcs), obj=obj, class_dict=class_funcs)
        log.info("# C3")
    else:
        log.info("C2: %s" % obj)
        StockPickler.save_global(pickler, obj)
        log.info("# C2")
    return


@register(ItemGetterType)
def save_itemgetter(pickler, obj):
    log.info("Ig: %s" % obj)
    helper = _itemgetter_helper()
    obj(helper)
    pickler.save_reduce(type(obj), tuple(helper.items), obj=obj)
    log.info("# Ig")
    return


@register(AttrGetterType)
def save_attrgetter(pickler, obj):
    log.info("Ag: %s" % obj)
    attrs = []
    helper = _attrgetter_helper(attrs)
    obj(helper)
    pickler.save_reduce(type(obj), tuple(attrs), obj=obj)
    log.info("# Ag")
    return


@register(PartialType)
def save_functor(pickler, obj):
    log.info("Fu: %s" % obj)
    pickler.save_reduce(_create_ftype, (type(obj), obj.func, obj.args,
                                        obj.keywords), obj=obj)
    log.info("# Fu")
    return


@register(SuperType)
def save_functor(pickler, obj):
    log.info("Su: %s" % obj)
    pickler.save_reduce(super, (obj.__thisclass__, obj.__self__), obj=obj)
    log.info("# Su")
    return


@register(BuiltinMethodType)
def save_builtin_method(pickler, obj):
    if obj.__self__ is not None:
        if obj.__self__ is __builtin__:
            module = 'builtins' if PY3 else '__builtin__'
            _t = "B1"
            log.info("%s: %s" % (_t, obj))
        else:
            module = obj.__self__
            _t = "B3"
            log.info("%s: %s" % (_t, obj))
        if is_dill(pickler):
            _recurse = pickler._recurse
            pickler._recurse = False
        pickler.save_reduce(_get_attr, (module, obj.__name__), obj=obj)
        if is_dill(pickler):
            pickler._recurse = _recurse
        log.info("# %s" % _t)
    else:
        log.info("B2: %s" % obj)
        StockPickler.save_global(pickler, obj)
        log.info("# B2")
    return


@register(MethodType)
def save_instancemethod0(pickler, obj):  # example: cStringIO.StringI
    log.info("Me: %s" % obj)             # XXX: obj.__dict__ handled elsewhere?

    args = (get_method_function(obj), get_method_self(obj)) if PY3 \
        else (get_method_function(obj), get_method_self(obj), obj.im_class)

    pickler.save_reduce(MethodType, args, obj=obj)

    log.info("# Me")

if sys.hexversion >= 0x20500f0:
    @register(MemberDescriptorType)
    @register(GetSetDescriptorType)
    @register(MethodDescriptorType)
    @register(WrapperDescriptorType)
    def save_wrapper_descriptor(pickler, obj):
        log.info("Wr: %s" % obj)
        pickler.save_reduce(_getattr, (obj.__objclass__, obj.__name__,
                                       obj.__repr__()), obj=obj)
        log.info("# Wr")
        return

    @register(MethodWrapperType)
    def save_instancemethod(pickler, obj):
        log.info("Mw: %s" % obj)
        pickler.save_reduce(getattr, (obj.__self__, obj.__name__), obj=obj)
        log.info("# Mw")
        return
else:
    @register(MethodDescriptorType)
    @register(WrapperDescriptorType)
    def save_wrapper_descriptor(pickler, obj):
        log.info("Wr: %s" % obj)
        pickler.save_reduce(_getattr, (obj.__objclass__, obj.__name__,
                                       obj.__repr__()), obj=obj)
        log.info("# Wr")
        return


@register(SliceType)
def save_slice(pickler, obj):
    log.info("Sl: %s" % obj)
    pickler.save_reduce(slice, (obj.start, obj.stop, obj.step), obj=obj)
    log.info("# Sl")
    return


@register(XRangeType)
@register(EllipsisType)
@register(NotImplementedType)
def save_singleton(pickler, obj):
    log.info("Si: %s" % obj)
    pickler.save_reduce(_eval_repr, (obj.__repr__(),), obj=obj)
    log.info("# Si")
    return

# thanks to Paul Kienzle for pointing out ufuncs didn't pickle
if NumpyArrayType:
    @register(NumpyUfuncType)
    def save_numpy_ufunc(pickler, obj):
        log.info("Nu: %s" % obj)
        StockPickler.save_global(pickler, obj)
        log.info("# Nu")
        return


@register(ReferenceType)
def save_weakref(pickler, obj):
    refobj = obj()
    log.info("R1: %s" % obj)
    pickler.save_reduce(_create_weakref, (refobj,), obj=obj)
    log.info("# R1")
    return


@register(ProxyType)
@register(CallableProxyType)
def save_weakproxy(pickler, obj):
    refobj = _locate_object(_proxy_helper(obj))
    try:
        _t = "R2"
        log.info("%s: %s" % (_t, obj))
    except ReferenceError:
        _t = "R3"
        log.info("%s: %s" % (_t, sys.exc_info()[1]))
    if type(obj) is CallableProxyType:
        callable = True
    else:
        callable = False
    pickler.save_reduce(_create_weakproxy, (refobj, callable), obj=obj)
    log.info("# %s" % _t)
    return


@register(ModuleType)
def save_module(pickler, obj):
    # if a module file name starts with prefx, it should be a builtin
    # module, so should be pickled as a reference

    if hasattr(obj, "__file__"):
        names = ["base_prefix", "base_exec_prefix", "exec_prefix",
                 "prefix", "real_prefix"]
        builtin_mod = any([obj.__file__.startswith(getattr(sys, name))
                           for name in names if hasattr(sys, name)])
        builtin_mod = builtin_mod or 'site-packages' in obj.__file__
        lib_path = abspath(join(sys.prefix, 'lib'))
        builtin_mod = builtin_mod or commonprefix([lib_path, obj.__file__]) == lib_path
    else:
        builtin_mod = True

    if obj.__name__ not in ("builtins", "dill") \
            and not builtin_mod or is_dill(pickler) and obj is pickler._main :
        log.info("M1: %s" % obj)
        pickler.save_reduce(_import_module, (obj.__name__,), obj=obj, state=obj.__dict__)
        log.info("# M1")
    else:
        log.info("M2: %s" % obj)
        pickler.save_reduce(_import_module, (obj.__name__,), obj=obj)
        log.info("# M2")
    return


@register(TypeType)
def save_type(pickler, obj):
    if obj in _type_map:
        log.info("T1: %s" % obj)
        pickler.save_reduce(_load_type, (_type_map[obj],), obj=obj)
        log.info("# T1")
    elif obj.__module__ == '__main__':
        try:  # use StockPickler for special cases [namedtuple,]
            [getattr(obj, attr) for attr in ('_fields', '_asdict',
                                             '_make', '_replace')]
            log.info("T6: %s" % obj)
            StockPickler.save_global(pickler, obj)
            log.info("# T6")
            return
        except AttributeError:
            pass
        if issubclass(type(obj), type):
            #   try: # used when pickling the class as code (or the interpreter)
            if is_dill(pickler) and not pickler._byref:
                # thanks to Tom Stepleton pointing out pickler._session unneeded
                _t = 'T2'
                log.info("%s: %s" % (_t, obj))
                _dict = _dict_from_dictproxy(obj.__dict__)
            else:
                log.info("T5: %s" % obj)
                StockPickler.save_global(pickler, obj)
                log.info("# T5")
                return
        else:
            _t = 'T3'
            log.info("%s: %s" % (_t, obj))
            _dict = obj.__dict__
        pickler.save_reduce(_create_type, (type(obj), obj.__name__,
                                           obj.__bases__, _dict), obj=obj)
        log.info("# %s" % _t)
    # special cases: NoneType
    elif isinstance(obj, type(None)):
        log.info("T7: %s" % obj)
        pickler.write_lines('c__builtin__\nNoneType\n')
        log.info("# T7")
    else:
        log.info("T4: %s" % obj)
        StockPickler.save_global(pickler, obj)
        log.info("# T4")
    return


@register(property)
def save_property(pickler, obj):
    log.info("Pr: %s" % obj)
    pickler.save_reduce(property, (obj.fget, obj.fset, obj.fdel, obj.__doc__), obj=obj)
    log.info("# Pr")


@register(staticmethod)
@register(classmethod)
def save_classmethod(pickler, obj):
    log.info("Cm: %s" % obj)
    im_func = '__func__' if PY3 else 'im_func'
    try:
        orig_func = getattr(obj, im_func)
    except AttributeError:  # Python 2.6
        orig_func = obj.__get__(None, object)
        if isinstance(obj, classmethod):
            orig_func = getattr(orig_func, im_func)  # Unbind
    pickler.save_reduce(type(obj), (orig_func,), obj=obj)
    log.info("# Cm")


# quick sanity checking
def pickles(obj, exact=False, safe=False, **kwds):
    """quick check if object pickles with dill"""
    if safe:
        exceptions = (Exception,)  # RuntimeError, ValueError
    else:
        exceptions = (TypeError, AssertionError, PicklingError, UnpicklingError)
    try:
        pik = copy(obj, **kwds)
        try:
            result = bool(pik.all() == obj.all())
        except AttributeError:
            result = pik == obj
        if result: return True
        if not exact:
            result = type(pik) == type(obj)
            if result: return result
            # class instances might have been dumped with byref=False
            return repr(type(pik)) == repr(type(obj))  # XXX: InstanceType?
        return False
    except exceptions:
        return False


def check(obj, *args, **kwds):
    """check pickling of an object across another process"""
    # == undocumented ==
    # python -- the string path or executable name of the selected python
    # verbose -- if True, be verbose about printing warning messages
    # all other args and kwds are passed to dill.dumps
    verbose = kwds.pop('verbose', False)
    python = kwds.pop('python', None)
    if python is None:
        import sys
        python = sys.executable
    # type check
    isinstance(python, str)
    import subprocess
    fail = True
    try:
        _obj = dumps(obj, *args, **kwds)
        fail = False
    finally:
        if fail and verbose:
            print("DUMP FAILED")
    msg = "%s -c import dill; print(dill.loads(%s))" % (python, repr(_obj))
    msg = "SUCCESS" if not subprocess.call(msg.split(None, 2)) else "LOAD FAILED"
    if verbose:
        print(msg)
    return


# use to protect against missing attributes
def is_dill(pickler):
    """check the dill-ness of your pickler"""
    return 'dill' in pickler.__module__
    # return hasattr(pickler,'_main')


def _extend():
    """extend pickle with all of dill's registered types"""
    # need to have pickle not choke on _main_module?  use is_dill(pickler)
    for t, func in Pickler.dispatch.items():
        try:
            StockPickler.dispatch[t] = func
        except (TypeError, PicklingError, UnpicklingError):
            log.info("skip: %s" % t)

