def num_to_label(num):
    alphabets = "0123456789' "
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret
