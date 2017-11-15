SYMBOL_MAP = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        '<unk>': 3
        }


def get_symbol_id(symbol):
    return SYMBOL_MAP[symbol]
