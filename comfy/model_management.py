"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import psutil
import logging
from enum import Enum
from comfy.cli_args import args
import torch
import sys
import importlib
import weakref
import gc

class VRAMState(Enum):
    DISABLED = 0
    NORMAL_VRAM = 1
    SHARED = 2

class CPUState(Enum):
    GPU = 0
    CPU = 1

# Steam Deck Configuration:
# We are always on an AMD GPU with shared memory.
vram_state = VRAMState.SHARED
cpu_state = CPUState.GPU
total_vram = 0

torch_version = ""
try:
    torch_version = torch.version.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
except:
    pass

if args.deterministic:
    logging.info("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

def is_intel_xpu():
    return False

def is_ascend_npu():
    return False

def is_mlu():
    return False

def is_ixuca():
    return False

def get_torch_device():
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        # On Steam Deck, we always use the ROCm/HIP device, which PyTorch sees as "cuda"
        return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
    if dev is None:
        dev = get_torch_device()

    if dev.type == 'cpu':
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        stats = torch.cuda.memory_stats(dev)
        mem_reserved = stats['reserved_bytes.all.current']
        _, mem_total_cuda = torch.cuda.mem_get_info(dev)
        mem_total_torch = mem_reserved
        mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))
logging.info("pytorch version: {}".format(torch_version))

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    return False

def is_amd():
    return torch.version.hip is not None

def amd_min_version(device=None, min_rdna_version=0):
    if not is_amd():
        return False

    if is_device_cpu(device):
        return False

    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.startswith('gfx') and len(arch) == 7:
        try:
            cmp_rdna_version = int(arch[4]) + 2
        except:
            cmp_rdna_version = 0
        if cmp_rdna_version >= min_rdna_version:
            return True

    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4

# Simplified for Steam Deck. PyTorch attention is a good default.
ENABLE_PYTORCH_ATTENTION = True
if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

PRIORITIZE_FP16 = True
torch.backends.cuda.matmul.allow_fp16_accumulation = True
logging.info("Enabled fp16 accumulation.")

try:
    if torch_version_numeric >= (2, 5):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except:
    logging.warning("Warning, could not set allow_fp16_bf16_reduction_math_sdp")

# VRAM state is simplified as we are always on a shared memory GPU.
logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = False

def get_torch_device_name(device):
    if device.type == "cuda":
        try:
            allocator_backend = torch.cuda.get_allocator_backend()
        except:
            allocator_backend = ""
        return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
    else:
        return "{}".format(device.type)

try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")

current_loaded_models = []

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
        real_model = self.model.model

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break

def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024

def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()

def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not DISABLE_SMART_MEMORY:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.SHARED: # Simplified from HIGH_VRAM
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    cleanup_models_gc()

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models_temp = set()
    for m in models:
        models_temp.add(m)
        for mm in m.model_patches_models():
            models_temp.add(mm)

    models = models_temp

    models_to_load = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            current_loaded_models.pop(i).model.detach(unpatch_all=False)

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < minimum_memory_required:
                models_l = free_memory(minimum_memory_required, device)
                logging.info("{} models unloaded.".format(len(models_l)))

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if (vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory

            lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)
    return

def load_model_gpu(model):
    return load_models_gpu([model])

def loaded_models(only_currently_used=False):
    output = []
    for m in current_loaded_models:
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output


def cleanup_models_gc():
    do_gc = False
    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if cur.is_dead():
            logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            do_gc = True
            break

    if do_gc:
        gc.collect()
        soft_empty_cache()

        for i in range(len(current_loaded_models)):
            cur = current_loaded_models[i]
            if cur.is_dead():
                logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))



def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size

def unet_offload_device():
    return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    return get_torch_device()

def maximum_vram_for_weights(device=None):
    return (get_total_memory(device) * 0.88 - minimum_inference_memory())

def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32], weight_dtype=None):
    # Hardcoded for Steam Deck (RDNA2) - FP16 is optimal
    if torch.float16 in supported_dtypes:
        return torch.float16
    return torch.float32

def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32 or weight_dtype == torch.float64:
        return None
    if weight_dtype == torch.float16:
        return None

    if torch.float16 in supported_dtypes:
        return torch.float16
    return torch.float32

def text_encoder_offload_device():
    return torch.device("cpu")

def text_encoder_device():
    return get_torch_device()

def text_encoder_initial_device(load_device, offload_device, model_size=0):
    return offload_device

def text_encoder_dtype(device=None):
    # Hardcoded for Steam Deck - FP16 is optimal
    return torch.float16

def intermediate_device():
    return torch.device("cpu")

def vae_device():
    return get_torch_device()

def vae_offload_device():
    return torch.device("cpu")

def vae_dtype(device=None, allowed_dtypes=[]):
    # Hardcoded for Steam Deck - RDNA2 has issues with BF16 VAE, FP16 is better
    if torch.float16 in allowed_dtypes:
        return torch.float16
    return torch.float32

def get_autocast_device(dev):
    return "cuda"

def supports_dtype(device, dtype):
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True # RDNA2 supports it, even if not always performant
    return False

def supports_cast(device, dtype):
    if dtype == torch.float32 or dtype == torch.float16 or dtype == torch.bfloat16:
        return True
    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype
    if not supports_cast(device, dtype):
        dtype = fallback_dtype
    return dtype

def device_supports_non_blocking(device):
    return True

def device_should_use_non_blocking(device):
    return False

def force_channels_last():
    return False

STREAMS = {}
NUM_STREAMS = 1

stream_counters = {}
def get_offload_stream(device):
    return None

def sync_stream(device, stream):
    return

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)

def xformers_enabled():
    return False

def xformers_enabled_vae():
    return False

def pytorch_attention_enabled():
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_enabled_vae():
    return False  # Known to cause issues on AMD

def pytorch_attention_flash_attention():
    return ENABLE_PYTORCH_ATTENTION

def force_upcast_attention_dtype():
    return None

def get_free_memory(dev=None, torch_free_too=False):
    if dev is None:
        dev = get_torch_device()

    if dev.type == 'cpu':
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        stats = torch.cuda.memory_stats(dev)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def cpu_mode():
    return cpu_state == CPUState.CPU

def mps_mode():
    return False

def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False

def is_device_cpu(device):
    return is_device_type(device, 'cpu')

def is_device_mps(device):
    return False

def is_device_xpu(device):
    return False

def is_device_cuda(device):
    return is_device_type(device, 'cuda')

def is_directml_enabled():
    return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if is_device_cpu(device):
        return False
    return True

def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    return False # RDNA2 has poor bf16 performance

def supports_fp8_compute(device=None):
    return False

def extended_fp16_support():
    return True

def soft_empty_cache(force=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def unload_all_models():
    free_memory(1e30, get_torch_device())

#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
