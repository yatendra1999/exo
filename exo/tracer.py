import inspect
import pickle
from inspect import FrameInfo
from torch import nn as torch_nn
from flax import nnx as flax_nnx
from flax import linen as linen_nn
from time import time_ns

from sqlalchemy import Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
sql_engine = create_engine("sqlite:///tracer.db", echo=True)
Session = sessionmaker(bind=sql_engine)

class TracerInstance(Base):
    __tablename__ = 'traced_calls'

    timestamp = Column(Integer, primary_key=True)  # Use timestamp as the primary key.
    exec_path = Column(String, nullable=False)
    data_dump = Column(LargeBinary, nullable=False)

    def __init__(self, exec_path: str, timestamp: int, data_dump: bytes):
        self.exec_path = exec_path
        self.timestamp = timestamp
        self.data_dump = data_dump

    def decode_args(self):
        return pickle.loads(self.data_dump)

Base.metadata.create_all(sql_engine)

def is_primitive(obj):
    return isinstance(obj, (int, float, str, bool, tuple, frozenset, bytes, complex))

def is_instance_of_user_defined_class(obj):
    return not is_primitive(obj) and hasattr(obj, "__class__") and not isinstance(obj, type)


def check_framework_instances(obj) -> str | None:
    if isinstance(obj, torch_nn.Module):
        return "torch"
    if isinstance(obj, flax_nnx.Module):
        return "flax.nnx"
    if isinstance(obj, linen_nn.Module):
        return "flax.linen"
    return None

def get_stack_name(stack: list[FrameInfo]) -> str:
    frame_stack = []
    for frame_info in stack:
        frame = frame_info.frame
        # Get the local variables in the frame
        local_vars = frame.f_locals
        
        # Check if 'self' is in local variables, indicating an object method
        if 'self' in local_vars:
            obj = local_vars['self']
            class_name = obj.__class__.__name__
            if not (len(frame_stack) > 0 and class_name == frame_stack[-1]):
                frame_stack.append(class_name)
    return '.'.join(frame_stack)

def hook_function(fn: callable) -> callable:

    def wrapped(*args, **kwargs):
        # Capture call stack.
        call_stack: list[FrameInfo] = inspect.stack()
        stack_name = get_stack_name(call_stack)

        # Serialize data.
        data = {
            "args": args,
            "kwargs": kwargs,
        }

        # Call the original function and capture return value.
        ret_val = fn(*args, **kwargs)
        data['ret'] = ret_val

        # Serialize to bytes.
        serialized_data = pickle.dumps(data)

        # Create a TracerInstance.
        tracer_instance = TracerInstance(
            exec_path=stack_name,
            timestamp=time_ns(),
            data_dump=serialized_data,
        )
        session = Session()
        # Insert into the database.
        session.add(tracer_instance)
        session.commit()

        return ret_val

    return wrapped

def hook_framework_module(obj, framework: str) -> None:
    fn_map = {
        "torch": "forward",
        "flax.nnx": "__call__",
        "flax.linen": "__call__"
    }
    if framework not in fn_map:
        return
    fn_name = fn_map[framework]
    if not hasattr(obj, fn_name):
        return
    potential_fn = getattr(obj, fn_name)
    if not callable(potential_fn):
        return
    setattr(obj, fn_name, hook_function(potential_fn))

def model_tracer(obj):
    if isinstance(obj, list):
        return [model_tracer(x) for x in obj]
    if isinstance(obj, dict):
        return { k: model_tracer(obj[k]) for k in obj}
    if inspect.isfunction(obj):
        return hook_function(obj)
    framework = check_framework_instances(obj)
    if framework is not None:
        hook_framework_module(obj, framework)
    if hasattr(obj, '__dict__'):
        attr_dict = obj.__dict__
        for attr in attr_dict:
            potential_obj = attr_dict[attr]
            if check_framework_instances(potential_obj) != None or isinstance(potential_obj, dict) or isinstance(potential_obj, list):
                model_tracer(potential_obj)

def load_from_timestamp(timestamp: int):
    session = Session()
    # Query all instances.
    instance = session.query(TracerInstance).where(TracerInstance.timestamp == timestamp).one()
    session.expunge(instance)
    session.close()
    pickle_data = pickle.loads(instance.data_dump)
    return instance, pickle_data