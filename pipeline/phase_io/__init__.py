"""Save/load CONDITIONING pairs and LATENTs to explicit file paths, so one generation can be split into
separate ComfyUI processes (encode -> high-noise -> low-noise -> decode). Process exit is the only reliable
way to give RAM back, so each phase's peak is its own instead of the sum of all models."""
import torch


class PhaseSaveCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "phase_io"

    def run(self, positive, negative, path):
        torch.save({"positive": positive, "negative": negative}, path)
        return ()


class PhaseLoadCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "run"
    CATEGORY = "phase_io"

    def run(self, path):
        d = torch.load(path, weights_only=False)
        return (d["positive"], d["negative"])


class PhaseSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "phase_io"

    def run(self, samples, path):
        torch.save({k: v for k, v in samples.items()}, path)
        return ()


class PhaseLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "phase_io"

    def run(self, path):
        return (torch.load(path, weights_only=False),)


class AnyType(str):
    """matches every socket type, so one save/load pair carries IMAGE batches and INTs (trim counts) alike"""
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class PhaseSaveAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY,), "path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "phase_io"

    def run(self, value, path):
        torch.save(value, path)
        return ()


class PhaseLoadAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}
    RETURN_TYPES = (ANY,)
    FUNCTION = "run"
    CATEGORY = "phase_io"

    def run(self, path):
        return (torch.load(path, weights_only=False),)


NODE_CLASS_MAPPINGS = {"PhaseSaveCond": PhaseSaveCond, "PhaseLoadCond": PhaseLoadCond,
                       "PhaseSaveLatent": PhaseSaveLatent, "PhaseLoadLatent": PhaseLoadLatent,
                       "PhaseSaveAny": PhaseSaveAny, "PhaseLoadAny": PhaseLoadAny}
