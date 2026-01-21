from __future__ import annotations

import collections.abc as cabc
import inspect
import typing as t
from dataclasses import dataclass

import gradio as gr

NoneType = type(None)


def _origin(tp: t.Any) -> t.Any:
    return t.get_origin(tp)


def _args(tp: t.Any) -> tuple[t.Any, ...]:
    return t.get_args(tp)


def _is_optional(tp: t.Any) -> bool:
    o = _origin(tp)
    if o is t.Union:
        a = _args(tp)
        return any(x is NoneType for x in a)
    return False


def _unwrap_optional(tp: t.Any) -> t.Any:
    if not _is_optional(tp):
        return tp
    return next(x for x in _args(tp) if x is not NoneType)


def _is_literal(tp: t.Any) -> bool:
    return _origin(tp) is t.Literal


def _is_enum(tp: t.Any) -> bool:
    try:
        import enum

        return isinstance(tp, type) and issubclass(tp, enum.Enum)
    except Exception:
        return False


def _is_str_stream_type(tp: t.Any) -> bool:
    if tp is inspect._empty:
        return False

    tp = _unwrap_optional(tp)
    o = _origin(tp)
    a = _args(tp)

    iterable_origins = {
        t.Iterable,
        t.Iterator,
        t.Generator,
        cabc.Iterable,
        cabc.Iterator,
        cabc.Generator,
    }

    if o in iterable_origins:
        if not a:
            return False
        # Generator[str, SendType, ReturnType] -> first arg is yield type
        return a[0] is str

    return False


def _is_runtime_str_stream(value: t.Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return False
    return isinstance(value, (cabc.Iterator, cabc.Generator))

@dataclass
class BuiltInput:
    component: gr.components.Component
    name: str


def _build_input_component(
    name: str,
    annotation: t.Any,
    default: t.Any,
) -> gr.components.Component:
    """
    Map a parameter to a Gradio component.
    """
    if annotation is inspect._empty:
        annotation = str

    ann = _unwrap_optional(annotation)

    # Literal -> Dropdown
    if _is_literal(ann):
        choices = list(_args(ann))
        value = default if default is not inspect._empty else (choices[0] if choices else None)
        return gr.Dropdown(choices=choices, value=value, label=name)

    # Enum -> Dropdown
    if _is_enum(ann):
        choices = list(ann)
        value = default if default is not inspect._empty else (choices[0] if choices else None)
        return gr.Dropdown(choices=choices, value=value, label=name, type="value")

    # bool -> Checkbox
    if ann is bool:
        value = default if default is not inspect._empty else False
        return gr.Checkbox(value=bool(value), label=name)

    # int/float -> Number
    if ann is int:
        value = default if default is not inspect._empty else 0
        return gr.Number(value=int(value), label=name, precision=0)

    if ann is float:
        value = default if default is not inspect._empty else 0.0
        return gr.Number(value=float(value), label=name)

    # str -> Textbox
    if ann is str:
        value = default if default is not inspect._empty else ""
        return gr.Textbox(value=value, label=name)

    # list/dict -> JSON
    if ann in (dict, list) or _origin(ann) in (dict, list):
        value = default if default is not inspect._empty else None
        return gr.JSON(value=value, label=name)

    # Fallback
    value = default if default is not inspect._empty else ""
    return gr.Textbox(value=value, label=name)


def _build_output_components(return_annotation: t.Any) -> list[gr.components.Component]:
    if return_annotation is inspect._empty or return_annotation is None or return_annotation is NoneType:
        return []

    ann = _unwrap_optional(return_annotation)
    o = _origin(ann)

    if o in (tuple, t.Tuple):
        outs: list[gr.components.Component] = []
        for i, sub in enumerate(_args(ann)):
            built = _build_output_components(sub)
            outs.extend(built or [gr.Textbox(label=f"output_{i}")])
        return outs

    if _is_str_stream_type(ann):
        return [gr.Textbox(label="result")]

    if ann is str:
        return [gr.Textbox(label="result")]
    if ann is bool:
        return [gr.Checkbox(label="result")]
    if ann is int:
        return [gr.Number(label="result", precision=0)]
    if ann is float:
        return [gr.Number(label="result")]
    if ann in (dict, list) or o in (dict, list):
        return [gr.JSON(label="result")]

    return [gr.Textbox(label="result")]


def create_gradio_interface(
    fn: t.Callable[..., t.Any],
    *,
    title: str | None = None,
    description: str | None = None,
    submit_text: str = "Submit",
    show_api: bool = False,
    server_name: str | None = None,
    server_port: int | None = None,
    share: bool = False,
    stream_append: bool = True,
) -> gr.Blocks:
    sig = inspect.signature(fn)
    hints = t.get_type_hints(fn)

    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"{fn.__name__} uses *args/**kwargs; not supported for auto UI.")

    params = list(sig.parameters.values())
    inputs: list[BuiltInput] = []
    for p in params:
        ann = hints.get(p.name, p.annotation)
        comp = _build_input_component(p.name, ann, p.default)
        inputs.append(BuiltInput(component=comp, name=p.name))

    ret_ann = hints.get("return", sig.return_annotation)
    outputs = _build_output_components(ret_ann)

    header = title or fn.__name__
    desc = description if description is not None else (inspect.getdoc(fn) or "")

    def _runner(*ui_args):
        kwargs = {bi.name: ui_args[i] for i, bi in enumerate(inputs)}
        result = fn(**kwargs)

        if not outputs:
            return None

        if len(outputs) == 1 and _is_runtime_str_stream(result):
            acc = ""
            for chunk in result:
                chunk = "" if chunk is None else str(chunk)
                if stream_append:
                    acc += chunk
                    yield acc
                else:
                    yield chunk
            return

        if len(outputs) > 1 and not isinstance(result, tuple):
            return (result,) + (None,) * (len(outputs) - 1)

        return result

    with gr.Blocks(title=header) as demo:
        gr.Markdown(f"# {header}")
        if desc:
            gr.Markdown(desc)

        with gr.Row():
            with gr.Column(scale=1):
                for bi in inputs:
                    bi.component.render()
                btn = gr.Button(submit_text)
            with gr.Column(scale=1):
                if outputs:
                    for out in outputs:
                        out.render()
                else:
                    gr.Markdown("_(No return value to display)_")

        btn.click(
            fn=_runner,
            inputs=[bi.component for bi in inputs],
            outputs=outputs if outputs else [],
            api_name=fn.__name__ if show_api else None,
        )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share
    )
    return demo
