from collections.abc import MutableSequence, Sequence

import jax
import numpy as np
from jax._src import mesh_utils

from sgl_jax.srt.utils.jax_utils import get_device_id_offset

default_mesh_axes = [
    "data",  # data parallelism
    "tensor",  # tensor parallelism
]


def _device_summary(devices) -> str:
    """Short human summary of available devices, e.g. '3 x tpu-v5e-pcie-8'."""
    counts: dict[str, int] = {}
    for dev in devices:
        key = f"{getattr(dev, 'platform', '?')} ({getattr(dev, 'device_kind', '?')})"
        counts[key] = counts.get(key, 0) + 1
    return "; ".join(f"{n} x {k}" for k, n in sorted(counts.items())) or "none"


def _validate_device_mesh_shape(
    mesh_shape: Sequence[int],
    num_devices: int,
    devices,
    requested_by: str,
) -> None:
    """Raise a clear, actionable error before JAX's cryptic mesh assertion."""
    required = int(np.prod(mesh_shape, dtype=int))
    if num_devices == required:
        return
    raise ValueError(
        f"Cannot create device mesh of shape {tuple(mesh_shape)} "
        f"({requested_by} needs {required} device(s)): only {num_devices} device(s) "
        f"available ({_device_summary(devices)}).\n"
        f"Remedies:\n"
        f"  * If you intend to serve on a TPU, make sure a TPU accelerator is "
        f"attached and that TPU_ACCELERATOR_TYPE / TPU_WORKER_HOSTNAMES are set "
        f"(e.g. enable TPU in the runtime/VM);\n"
        f"  * Otherwise reduce the requested parallelism to match the available "
        f"device count (e.g. --tp-size {num_devices} --dp-size 1), or run on a "
        f"non-TPU backend with --device cpu."
    )


def create_device_mesh(
    ici_parallelism: MutableSequence[int],
    dcn_parallelism: MutableSequence[int],
    devices=None,
    device_indexes: list[int] | None = None,
    num_slices: int = 1,
    allow_split_physical_axes: bool = True,
    use_explicit_sharding: bool = True,
    mesh_axes: Sequence[str] = default_mesh_axes,
) -> jax.sharding.Mesh:
    """Create a device mesh"""
    if devices is None:
        devices = jax.devices()

    offset = get_device_id_offset(devices)

    if device_indexes is not None:
        device_indexes = [idx + offset for idx in device_indexes]
        devices_dict = {device.id: device for device in devices}
        devices = [devices_dict.get(i) for i in list(set(device_indexes))]
    ici_parallelism = fill_unspecified_parallelism(ici_parallelism, len(devices))
    if num_slices > 1:
        dcn_parallelism = fill_unspecified_parallelism(dcn_parallelism, num_slices)
        _validate_device_mesh_shape(
            mesh_shape=ici_parallelism + dcn_parallelism,
            num_devices=len(devices),
            devices=devices,
            requested_by=f"ici_parallelism x dcn_parallelism (num_slices={num_slices})",
        )
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism,
            dcn_parallelism,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        all_devices = jax.devices()
        is_subset = len(devices) < len(all_devices)
        if is_subset:
            # JAX's create_device_mesh infers the full physical TPU topology
            # and asserts len(devices) == np.prod(dims), which fails when
            # only a subset of devices is used.  Fall back to a simple reshape.
            _validate_device_mesh_shape(
                mesh_shape=ici_parallelism,
                num_devices=len(devices),
                devices=devices,
                requested_by="ici_parallelism",
            )
            devices_array = np.array(devices).reshape(ici_parallelism)
        else:
            _validate_device_mesh_shape(
                mesh_shape=ici_parallelism,
                num_devices=len(devices),
                devices=devices,
                requested_by="ici_parallelism",
            )
            devices_array = mesh_utils.create_device_mesh(
                ici_parallelism,
                devices=devices,
                contiguous_submeshes=False,
                allow_split_physical_axes=allow_split_physical_axes,
            )

    if use_explicit_sharding:
        axis_types = (jax.sharding.AxisType.Explicit,) * len(mesh_axes)
        mesh = jax.sharding.Mesh(devices_array, mesh_axes, axis_types=axis_types)
    else:
        mesh = jax.sharding.Mesh(devices_array, mesh_axes)
    return mesh


def fill_unspecified_parallelism(
    parallelism: MutableSequence[int], num_devices: int
) -> MutableSequence[int]:
    if -1 not in parallelism:
        return parallelism

    assert parallelism.count(-1) == 1, "At most one axis can be unspecified."
    unspecified_axis_idx = parallelism.index(-1)
    determined_val = num_devices / np.prod(parallelism) * -1
    assert (
        determined_val >= 1 and determined_val.is_integer()
    ), "Unspecified value unable to be determined with the given parallelism values"
    parallelism[unspecified_axis_idx] = int(determined_val)
    return parallelism
