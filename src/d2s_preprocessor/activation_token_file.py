import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "data"


def get_vat_act_token_file_path():
    res = DATA_DIR / "activation_token" / "vat_activation_token.csv"
    if not res.exists():
        raise FileNotFoundError(
            f"File Token for VAT not found: {res}"
        )
    return res


def get_invoice_number_token_file_path():
    res = DATA_DIR / "activation_token" / "invoice_number_token.csv"
    if not res.exists():
        raise FileNotFoundError(
            f"File Token for Invoice Number not found {res}"
        )
    return res


def get_amounts_activation_token_path():
    res = DATA_DIR / "activation_token" / "amounts_activation_token.csv"
    raise FileNotFoundError(
        f"File Token for Amounts not found: {res}"
    )
    return res
