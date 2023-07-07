import abc
import dataclasses
from typing import Type, TypeVar, Generic, Iterable

import yaml


T = TypeVar("T", bound=Type)


class _SerializationHandler(Generic[T], metaclass=abc.ABCMeta):
    def __init__(self, handle_type: T, name: str=None):
        self.handle_type = handle_type
        self.name = name or f"!{handle_type.__name__}"

    @abc.abstractmethod
    def representer(self, dumper: yaml.SafeDumper, obj: T):
        pass

    @abc.abstractmethod
    def constructor(self, loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> T:
        pass


def _get_handlers(
        *,
        data_class_types: Iterable[Type] = tuple(),
        collection_types: Iterable[Type] = tuple(),
        mapping_types: Iterable[Type] = tuple(),
        enum_types: Iterable[Type] = tuple(),
):
    class CollectionClassHandler(_SerializationHandler[T]):
        def representer(self, dumper: yaml.SafeDumper, obj: T):
            return dumper.represent_sequence(self.name, obj)

        def constructor(self, loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> T:
            sequence = list(loader.construct_sequence(node))
            return self.handle_type(sequence)

    class MappingClassHandler(_SerializationHandler[T]):
        def representer(self, dumper: yaml.SafeDumper, obj: T):
            return dumper.represent_mapping(self.name, obj)

        def constructor(self, loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> T:
            sequence = loader.construct_mapping(node)
            return self.handle_type(sequence)

    class DataClassHandler(_SerializationHandler[T]):
        def representer(self, dumper: yaml.SafeDumper, obj: T):
            mapping = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
            return dumper.represent_mapping(self.name, mapping)

        def constructor(self, loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> T:
            mapping = dict(loader.construct_pairs(node))
            return self.handle_type(**mapping)

    class EnumHandler(_SerializationHandler[T]):
        def representer(self, dumper: yaml.SafeDumper, obj: T):
            return dumper.represent_scalar(self.name, obj.name)

        def constructor(self, loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> T:
            return self.handle_type(loader.construct_yaml_str(node))


    return [
        *(DataClassHandler(t) for t in data_class_types),
        *(CollectionClassHandler(t) for t in collection_types),
        *(MappingClassHandler(t) for t in mapping_types),
        *(EnumHandler(t) for t in enum_types),
    ]


def yaml_dumper_with_our_dataclasses(
        *,
        data_class_types: Iterable[Type] = tuple(),
        collection_types: Iterable[Type] = tuple(),
        mapping_types: Iterable[Type] = tuple(),
):
    safe_dumper = yaml.SafeDumper
    safe_dumper.ignore_aliases = lambda *args: True

    for handler in _get_handlers(
            data_class_types=data_class_types,
            collection_types=collection_types,
            mapping_types=mapping_types
     ):
        safe_dumper.add_representer(handler.handle_type, handler.representer)

    return safe_dumper


def yaml_loader_with_our_dataclasses(
        *,
        data_class_types: Iterable[Type] = tuple(),
        collection_types: Iterable[Type] = tuple(),
        mapping_types: Iterable[Type] = tuple(),
):
    loader = yaml.CSafeLoader
    for handler in _get_handlers(
            data_class_types=data_class_types,
            collection_types=collection_types,
            mapping_types=mapping_types,
    ):
        loader.add_constructor(handler.name, handler.constructor)
    return loader
