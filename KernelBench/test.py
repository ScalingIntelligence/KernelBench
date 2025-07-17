import torch

print("Before patch:", torch.randn)                 # built-in

import src.utils      # or: import src.utils
print("After  patch:", torch.randn)                 # <function _randn_patched at 0x…>

print("Module  :", torch.randn.__module__)          # should be 'src.utils'