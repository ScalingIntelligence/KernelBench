import ast

def is_generated_kernel_used(code: str) -> bool:
    class KernelUseAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.generated_kernel_vars = set()  # e.g., matmul_cuda
            self.generated_kernel_attrs = set()  # e.g., self.matmul_cuda
            self.generated_kernel_attrs_with_method = set()  # e.g., self.matmul_cuda.matmul_cpp
            self.overwritten_attrs = set()
            self.called_vars = set()
            self.called_attrs = set()
            self.called_attrs_with_method = set()
            self.pytorch_layer_names = {
                "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d",
                "ConvTranspose2d", "ConvTranspose3d", "Linear", "BatchNorm1d",
                "BatchNorm2d", "Embedding", "Sequential"
            }

        def visit_Assign(self, node):
            # Case 1: Detect kernel var from load_inline
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "load_inline":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.generated_kernel_vars.add(target.id)

            # Case 2: Track assignment to self.<attr> = <kernel var>
            if isinstance(node.value, ast.Name) and node.value.id in self.generated_kernel_vars:
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        self.generated_kernel_attrs.add(target.attr)
            
            # Case 2.5: Track assignment to self.<attr> = <kernel var>.<attr>
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.value.id in self.generated_kernel_vars:
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        self.generated_kernel_attrs_with_method.add(target.attr)

            # Case 3: Detect overwrites like self.attr = nn.Conv1d(...)
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr in self.pytorch_layer_names:
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            self.overwritten_attrs.add(target.attr)

            self.generic_visit(node)

        def visit_Call(self, node):
            # Case 1: Detect self.<attr>() call
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                self.called_attrs_with_method.add(node.func.attr)

            # Case 2: Detect self.<attr>.<method>() call
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                if isinstance(func.value.value, ast.Name) and func.value.value.id == "self":
                    attr_name = func.value.attr
                    self.called_attrs.add(attr_name)

            # Case 3: Detect <kernel_var>.<method>() call
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name): 
                if func.value.id in self.generated_kernel_vars:
                    self.called_vars.add(func.value.id)
                
            self.generic_visit(node)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, None, None, None, None, None  # Malformed Python code

    analyzer = KernelUseAnalyzer()
    analyzer.visit(tree)

    # If a generated kernel var was assigned to a self.attr and that attr was later used (and not overwritten)
    used_attrs = (analyzer.generated_kernel_attrs & analyzer.called_attrs) | (analyzer.generated_kernel_attrs_with_method & analyzer.called_attrs_with_method)
    effective_used_attrs = used_attrs - analyzer.overwritten_attrs
    used_vars = analyzer.generated_kernel_vars & analyzer.called_vars 

    is_used = len(effective_used_attrs) > 0 or len(used_vars) > 0
    return is_used


def torch_function_used(code: str) -> bool:
    """
    Checks whether the input code uses any torch function (e.g. torch.matmul) or nn layer (e.g. nn.Conv1d, nn.Linear, etc.)
    inside the ModelNew class only, but ignores usage of nn.Module itself.
    Returns True if any such usage is found in ModelNew, otherwise False.
    """
    import ast

    class TorchFunctionAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.torch_used = False
            self.nn_used = False
            self.torch_aliases = set()
            self.nn_aliases = set()
            self.nn_module_names = {"Module", "Parameter", "init"}  # ignore nn.Module
            self.torch_module_names = {"tensor", "Tensor", "ones", "zeros", "empty", "randn"}  # ignore nn.Module

        def visit_Import(self, node):
            for alias in node.names:
                if alias.name == "torch":
                    self.torch_aliases.add(alias.asname or "torch")
                if alias.name == "torch.nn":
                    self.nn_aliases.add(alias.asname or "nn")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == "torch":
                for alias in node.names:
                    self.torch_aliases.add(alias.asname or alias.name)
            if node.module == "torch.nn":
                for alias in node.names:
                    self.nn_aliases.add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # Check for torch.* or nn.* usage, but ignore nn.Module
            value = node.value
            if isinstance(value, ast.Name):
                if value.id in self.torch_aliases and not self.torch_used:
                    if node.attr not in self.torch_module_names:
                        # print(f"torch_used: {node.attr}")
                        self.torch_used = True
                if value.id in self.nn_aliases and not self.nn_used:
                    if node.attr not in self.nn_module_names:
                        # print(f"nn_used: {node.attr}")
                        self.nn_used = True
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check for torch.*() or nn.*() calls, but ignore nn.Module()
            func = node.func
            if isinstance(func, ast.Attribute):
                value = func.value
                if isinstance(value, ast.Name):
                    if value.id in self.torch_aliases and not self.torch_used:
                        if func.attr not in self.torch_module_names:
                            # print(f"torch_used: {func.attr}")
                            self.torch_used = True
                    if value.id in self.nn_aliases and not self.nn_used:
                        if func.attr not in self.nn_module_names:
                            # print(f"nn_used: {func.attr}")
                            self.nn_used = True
            self.generic_visit(node)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # First, collect import aliases globally
    import_analyzer = TorchFunctionAnalyzer()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_analyzer.visit(node)

    # Now, find the ModelNew class
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            # For each method in ModelNew, analyze its body
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    analyzer = TorchFunctionAnalyzer()
                    analyzer.torch_aliases = set(import_analyzer.torch_aliases)
                    analyzer.nn_aliases = set(import_analyzer.nn_aliases)
                    analyzer.visit(item)
                    if analyzer.torch_used or analyzer.nn_used:
                        return True
            return False  # No torch/nn usage found in ModelNew
    return False  # No ModelNew class found

