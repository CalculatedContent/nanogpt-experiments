__version__ = "0.1.0"

# WW-PGD is an installed dependency. Adapt to the public package API at runtime
# rather than requiring a repository-specific commit or fork.
from wwgpt._wwpgd_compat import install_wwpgd_api_compatibility, patch_wwgpt_ww_module

WWPGD_PROVENANCE = install_wwpgd_api_compatibility()

# Load and patch the adapter before train.py imports compatibility names such as
# WWPGD_COMMIT. That legacy name now contains observed runtime provenance only.
from wwgpt import ww as _ww

patch_wwgpt_ww_module(_ww, WWPGD_PROVENANCE)

del _ww
