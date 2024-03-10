import torch
import sys, os
import torch.nn.utils.prune as prune
import math


class PartRandomMethod(prune.BasePruningMethod):
    """
    Prune edges with magnitude under threshold
    """
    PRUNING_TYPE = 'unstructured'
    
    def __init__(self, amount: float, p: float):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.p = p

    def compute_mask(self, t: torch.Tensor, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:
            # larget=False -> bottom k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)

            # randomly sample from topk with a probability of p
            selected = torch.multinomial(input=topk.indices.to(dtype=torch.float32), num_samples=math.floor(nparams_toprune*self.p), replacement=False)
            index = topk.indices[selected]
            mask.view(-1)[index] = 0
        
        return mask
    
    @classmethod
    def apply(cls, module, name, amount, p, importance_scores=None):
        return super(PartRandomMethod, cls).apply(
            module, name, amount=amount, p=p, importance_scores=importance_scores
        )

class PartScaleMethod(prune.BasePruningMethod):
    """
    Scale the parameters with magnitude under threshold
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount: float, p: float):
        """
        Args: 
            p: scale the parameters with a ratio of p
        """
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.p = p
    
    def compute_mask(self, t: torch.Tensor, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if tensor_size != 0:
            # larget=False -> bottom k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            index = topk.indices
            mask.view(-1)[index] *= self.p

        return mask

    @classmethod
    def apply(cls, module, name, amount, p, importance_scores=None):
        return super(PartScaleMethod, cls).apply(
            module, name, amount=amount, p=p, importance_scores=importance_scores
        )

class AllScaleMethod(prune.BasePruningMethod):
    """
    Scale the parameters in a random way
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, p: float):
        self.p = p
    
    def compute_mask(self, t: torch.Tensor, default_mask):
        tensor_size = t.nelement()

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if tensor_size != 0:
            mask *= self.p

        return mask

    @classmethod
    def apply(cls, module, name, p, importance_scores=None):
        return super(AllScaleMethod, cls).apply(
            module, name, p=p, importance_scores=importance_scores
        )

def PartRandomUnstructured(module, name, amount, p):
    """
    ### Args:
        module: module to prune
        name: parameter name within `module` on which pruning will act.
        amount: decide the threshold
        p: randomly sample from topk with a ratio of p
    """
    PartRandomMethod.apply(module, name, amount, p)

def AllRandomUnstructured(module, name, amount):
    """
    ### Args:
        module: module to prune
        name: parameter name within `module` on which pruning will act.
        amount: decide the amount of parameters to prune
    """
    prune.RandomUnstructured.apply(module, name, amount)

def PartScaleUnstructured(module, name, amount, p):
    """
    ### Args:
        module: module to prune
        name: parameter name within `module` on which pruning will act.
        amount: decide the threshold
        p: scale the parameters with a ratio of p
    """
    PartScaleMethod.apply(module, name, amount, p)

def AllScaleUnstructured(module, name, p):
    """
    ### Args:
        module: module to prune
        name: parameter name within `module` on which pruning will act.
        amount: decide the amount of parameters to prune
        p: scale the parameters with a ratio of p
    """
    AllScaleMethod.apply(module, name, p)

def auto_file_name(path, file_name):
    basename, ext = os.path.splitext(file_name)
    i = 1
    new_name = file_name
    while os.path.exists(os.path.join(path, new_name)):
        new_name = f"{basename}({i}){ext}"
        i += 1
    return os.path.join(path, new_name)

save_path = "../results/test_torch_prune/"

def part_prune(model, prune_rate, p) -> torch.nn.Module:
    """
    Randomly prune the model
    
    Prune the input model's parameters of convolutional layer by their magnitudes.

    ### Args:
        model: Model to prune.
        prune_rate: Decides the pruning ratio.
        p: Decides the ratio of parameters selected to be applied to the mask.
    
    ### Returns:
        Pruned model.
    """
    list1 = [
        module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    ]
    parameters_to_prune = list1
    for module in parameters_to_prune:
        PartRandomUnstructured(
            module=module,
            name="weight",
            amount=prune_rate,
            p=p
        )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model

def all_prune(model, prune_rate) -> torch.nn.Module:
    """
    Randomly prune the model
    
    Prune the input model's parameters of convolutional layer in a random way.

    ### Args:
        model: Model to prune.
        prune_rate: Decides the pruning ratio.
    
    ### Returns:
        Pruned model.
    """
    list1 = [
        module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    ]
    parameters_to_prune = list1
    for module in parameters_to_prune:
        AllRandomUnstructured(
            module=module,
            name="weight",
            amount=prune_rate
        )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model

def part_scale(model, scale_rate, p) -> torch.nn.Module:
    """
    Scale the model
    
    Scale the input model's parameters of convolutional layer by their magnitudes.

    ### Args:
        model: Model to scale.
        scale_rate: Decides the scaling ratio.
        p: Decides the ratio of parameters selected to be applied to the mask.
    
    ### Returns:
        Scaled model.
    """
    list1 = [
        module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    ]
    parameters_to_prune = list1
    for module in parameters_to_prune:
        PartScaleUnstructured(
            module=module,
            name="weight",
            amount=scale_rate,
            p=p
        )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model

def all_scale(model, p) -> torch.nn.Module:
    """
    Scale the model
    
    Scale the input model's parameters of convolutional layer in a random way.

    ### Args:
        model: Model to scale.
        scale_rate: Decides the scaling ratio.
        p: Decides the ratio of parameters selected to be applied to the mask.
    
    ### Returns:
        Scaled model.
    """
    list1 = [
        module for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
    ]
    parameters_to_prune = list1
    for module in parameters_to_prune:
        AllScaleUnstructured(
            module=module,
            name="weight",
            p=p
        )
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model