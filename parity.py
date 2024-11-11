from constants import EVEN_PARITY, EVEN_PARITY_IDX, ODD_PARITY_IDX, ODD_PARITY

def parity_idx_to_parity(parity_idx: int) -> int:
    # return parity_idx // 2
    if parity_idx == EVEN_PARITY_IDX:
        return EVEN_PARITY
    else:
        return ODD_PARITY
    
    # if you wanted to remove the if statements and get the parity as fast as possible, you could do this:
    # return parity_idx*(-2) + 1
    #    when parity_idx is even, we are returning 0 + 1
    #    when parity_idx is odd, we are returning 1*(-2) + 1 = -1

def parity_for_l(l: int) -> int:
    if l % 2 == 0:
        return EVEN_PARITY
    else:
        return ODD_PARITY
    
def parity_to_parity_idx(parity: int) -> int:
    if parity == EVEN_PARITY:
        return EVEN_PARITY_IDX
    else:
        return ODD_PARITY_IDX