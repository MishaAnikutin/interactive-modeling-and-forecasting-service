from enum import Enum


class ZeroMethod(str, Enum):
    wilcox: str = "wilcox"
    zsplit: str = "zsplit"
    pratt: str = "pratt"
