import logging
import re
import nltk
import numpy as np
import pandas as pd
import pathlib

from d2s_image2text.paddle_ocr import TextExtractorPaddleOCR
from .utils.read_image import image_from_file


DATA_DIR = pathlib.Path(__file__).parent / "data"

_logger = logging.getLogger(__name__)


def word_similarity(word1, word2):
    w1 = set(word1)
    w2 = set(word2)
    return nltk.jaccard_distance(w1, w2)


def get_total_ht(row: dict):
    row = pd.Series(row, dtype=float)
    ttc_tokens = ["ht", "sous total"]  # Sous Total
    row_indexes = set(
        i
        for i in row.index
        for ttc_token in ttc_tokens
        for word in i.lower().replace(".", "").split(" ")
        if word_similarity(ttc_token, word) < 0.4
    )
    return row[list(row_indexes)].max()
    # return row[row.index.str.lower().str.replace(".", "", regex=False).str.contains("ht")].max()


def get_total_ttc(row: dict):
    return max(row.values()) if row.values() else None
    # row = pd.Series(row, dtype=float)
    # ttc_tokens = ["ttc", "net"]
    # row_indexes = set(i
    #                   for i in row.index
    #                   for ttc_token in ttc_tokens
    #                   for word in i.lower().replace(".", "").split(" ")
    #                   if word_similarity(ttc_token, word) < 0.4)
    # return row[list(row_indexes)].max()


def get_total_tva(row: dict):
    row = pd.Series(row, dtype=float)
    # return row[row.index.str.lower().str.replace(".", "", regex=False).str.contains("tva")].max()
    tva_tokens = ["tva", "19%", "19.00%", "19%"]
    row_indexes = set(
        i
        for i in row.index
        for tva_token in tva_tokens
        for word in i.lower().replace(".", "").split(" ")
        if word_similarity(tva_token, word) < 0.4
    )

    filtered_row = row[list(row_indexes)]
    # Retourne la plus petite valeur non null si elle existe et 0 si necessaire
    min_not_null = filtered_row[filtered_row > 0].min()
    return min_not_null if min_not_null else filtered_row.min()


def get_total_timbre(row: dict):
    row = pd.Series(row, dtype=float)
    timbre_tokens = ["timbre"]
    row_indexes = set(
        i
        for i in row.index
        for timbre_token in timbre_tokens
        for word in i.lower().replace(".", "").split(" ")
        if word_similarity(timbre_token, word) < 0.4
    )
    if row_indexes:
        return row[list(row_indexes)].min()
    return 1


def get_total_remise(row: dict):
    row = pd.Series(row, dtype=float)
    return row[
        row.index.str.lower().str.replace(".", "", regex=False).str.contains("remise")
    ].min()


def get_total_taxe(row: dict):
    row = pd.Series(row, dtype=float)
    taxe_tokens = ["taxes", "tax", "taxe"]
    row_indexes = set(
        i
        for i in row.index
        for token in taxe_tokens
        for word in i.lower().replace(".", "").split(" ")
        if word_similarity(token, word) < 0.4
    )
    return row[list(row_indexes)].min()
    # return row[row.index.str.lower().str.replace(".", "", regex=False).str.contains("tax")].min()


def ratio_similarity(
    box: pd.Series, token_df: pd.DataFrame, treshold: float = 0.4
) -> float:
    """
    Prend un box et calcule comment cette box est un label de montant
    @param treshold:
    @param box: La boite à vérifier si c'est un token d'activation de montant
    @param token_df: Un dataframe avec pour colonne [token, ratio].
                    C'est la liste des labels pour les montants avec leurs degrés d'importance (ratio)
    @return: Retourne la somme des ratios pour chaque token trouvé dans la boite (box)
    """
    ratio = 0
    for _, token_row in token_df.iterrows():
        is_similar = any(
            word_similarity(token_row.token.lower(), text) <= treshold
            for text in box.text.lower().strip().split(" ")
        )
        if token_row.token.lower() in box.text.lower() or is_similar:
            ratio += token_row.ratio
        if is_similar:
            _logger.debug(f"{token_row.token.lower()} = {box.text.lower().strip()}")
    return ratio


class InvoiceDataExtractor(object):
    """Classe de base pour extraire les informations dans une facture"""

    def __init__(self, image: np.ndarray, filename=None):
        self.image = image
        self.filename = filename

        self.__init_regex__()

        self.__init_activation_token__()
        self.extractor = TextExtractorPaddleOCR(image=image)
        self.bboxes = self.extractor.image2boxes()
        self.invoice_line_header = self._get_invoice_line_header()
        self.base_tax_line_header = self._get_base_tax_line_header()
        self.data = None

        self.caches = {"left_box": {}, "bottom_box": {}}

    def __init_regex__(self):
        self.vat_regex = r"(?P<pattern1>(\d{3,}[\w \/]+(000)?))"
        self.date_regex = (
            r"(?P<date>(\d{2}/\d{2}/\d{4})|"
            r"(\d{2}-\d{2}-\d{4})|"
            r"(\d{2}/\d{2}/\d{2})|"
            r"(\d{2}-\d{2}-\d{2})|"
            r"(\d{1,2}[-\/]{1}\d{1,2}[-\/]{1}\d{2}))"
        )
        self.invoice_number_regex = (
            r"(\s|^)(?P<pattern1>(F[ACT]*[ 0-9_\/]*\d{2,}))|"
            r"(?P<pattern2>[A-Z]*[0-9\-\/]{2,}\d{2,})(\s|$)"
        )

    def __init_activation_token__(self):
        vat_act_token_file_path = (
            DATA_DIR / "activation_token" / "vat_activation_token.csv"
        )
        if not vat_act_token_file_path.exists():
            raise FileNotFoundError(
                f"File Token for VAT not found: {vat_act_token_file_path}"
            )

        invoice_number_token_file_path = (
            DATA_DIR / "activation_token" / "invoice_number_token.csv"
        )
        if not invoice_number_token_file_path.exists():
            raise FileNotFoundError(
                f"File Token for Invoice Number not found {invoice_number_token_file_path}"
            )

        amounts_activation_token_path = (
            DATA_DIR / "activation_token" / "amounts_activation_token.csv"
        )
        if not amounts_activation_token_path.exists():
            raise FileNotFoundError(
                f"File Token for Amounts not found: {amounts_activation_token_path}"
            )

        self.vat_activation_token = pd.read_csv(vat_act_token_file_path)
        self.invoice_number_activation_token = pd.read_csv(
            invoice_number_token_file_path
        )
        self.amounts_activation_token = pd.read_csv(amounts_activation_token_path)

    def _get_invoice_line_header(self) -> pd.DataFrame:
        """
        Find invoice line header

        Recherche toute les boites de la même ligne et rassenble text
        Esuite utilise une liste de mots present courament dans les entêtes de ligne de
        ligne de facture pour donner une probabilité de si la boite est une boite d'entête de ligne de factur ou non
        """
        MIN_HEADER_COUNT = 3
        # Esuite utilise une liste de mots present courament dans les entêtes de ligne de
        list_product_def = ["Référence", "Produit", "Désignation", "Description"]
        list_qte_def = ["Qté", "QTE", "Quantité"]
        list_prix_def = [
            "PU",
            "P.U",
            "P.U.",
            "Unitaire",
            "Prix unitaire",
            "Px unitaire",
            "Prix unitaire",
        ]
        invoice_line_header_prod_df = pd.DataFrame(
            {"token": list_product_def, "ratio": [1 for i in list_product_def]}
        )
        invoice_line_header_qte_df = pd.DataFrame(
            {"token": list_qte_def, "ratio": [1 for i in list_qte_def]}
        )
        invoice_line_header_prix_df = pd.DataFrame(
            {"token": list_prix_def, "ratio": [1 for i in list_prix_def]}
        )
        header_df = pd.DataFrame()
        max_res = -1
        already_read = set()
        for index, box in self.bboxes.iterrows():
            if index in already_read:
                continue
            already_read.add(index)
            # Recherche toute les boites de la même ligne et rassenble text
            df = self.bboxes[
                self.bboxes.y0.between(box.y0, box.y1)
                | self.bboxes.y1.between(box.y0, box.y1)
            ]
            for i in df.index:
                already_read.add(i)
            res_prix = sum(
                ratio_similarity(row, invoice_line_header_prix_df)
                for i, row in df.iterrows()
            )
            res_prod = sum(
                ratio_similarity(row, invoice_line_header_prod_df)
                for i, row in df.iterrows()
            )
            res_qte = sum(
                ratio_similarity(row, invoice_line_header_qte_df)
                for i, row in df.iterrows()
            )

            if MIN_HEADER_COUNT <= len(df) and max_res < sum(
                (res_prix, res_prod, res_qte)
            ):
                header_df = df
                max_res = sum((res_prix, res_prod, res_qte))

        return header_df

    def __find_left_and_bottom_box__(self, vat_activation_token: pd.DataFrame):
        """

        @param vat_activation_token:
        @return:
        """
        left_bottom_bboxes = pd.DataFrame(columns=self.bboxes.columns)
        for _, token_row in vat_activation_token.iterrows():
            if token_row.ratio >= 1:
                for index, box in self.bboxes[
                    self.bboxes.text.str.lower().str.contains(token_row.token)
                ].iterrows():
                    left_bottom_bboxes = pd.concat(
                        [
                            left_bottom_bboxes,
                            self.bboxes[
                                self.bboxes.index == index
                            ],  # La boite elle meme
                            self.__get_left_box__(box),
                            self.__get_bottom_box__(box),
                        ],
                        axis=0,
                    )
        return left_bottom_bboxes

    def __get_left_box__(self, of, limit=3, sort_by="angle"):
        # # Check if exist in Cache
        # index = self.caches['left_box'].get(of.name, None)
        # if index is None:
        df = self.bboxes.copy()
        left_mask = (df.x0 > of.x1) & (
            ((of.y0 <= df.y0) & (df.y0 <= of.y1))
            | ((of.y0 <= df.y1) & (df.y1 <= of.y1))
            | ((df.y0 <= of.y0) & (of.y0 <= df.y1))
            | ((df.y0 <= of.y1) & (of.y1 <= df.y1))
        )
        df = df.loc[left_mask]
        df["angle"] = np.abs(
            np.arctan2(
                df.y0 + df.h / 2 - of.y0 - of.h / 2, df.x0 + df.w / 2 - of.x0 - of.w / 2
            )
        )
        df = df.sort_values(sort_by)
        # self.caches['left_box'][of.name] = index
        return df.iloc[:limit]

    def __get_right_box__(self, of, limit=3, sort_by="angle"):
        # # Check if exist in Cache
        # index = self.caches['left_box'].get(of.name, None)
        # if index is None:
        df = self.bboxes.copy()
        left_mask = (df.x0 > of.x1) & (
            ((of.y0 <= df.y0) & (df.y0 <= of.y1))
            | ((of.y0 <= df.y1) & (df.y1 <= of.y1))
            | ((df.y0 <= of.y0) & (of.y0 <= df.y1))
            | ((df.y0 <= of.y1) & (of.y1 <= df.y1))
        )
        df = df.loc[left_mask]
        df["angle"] = np.abs(
            np.arctan2(
                df.y0 + df.h / 2 - of.y0 - of.h / 2, df.x0 + df.w / 2 - of.x0 - of.w / 2
            )
        )
        df = df.sort_values(sort_by)
        # self.caches['left_box'][of.name] = index
        return df.iloc[:limit]

    def __get_bottom_box__(self, of, limit=1, sort_by="distance"):
        # # Check if exist in Cache
        # index = self.caches['bottom_box'].get(of.name, None)
        # if index is None:
        df = self.bboxes.copy()
        df["distance"] = np.sqrt((df.y0 - of.y0) ** 2 + (df.x0 - of.x0) ** 2)
        bottom_mask = (df.y0 > of.y1) & (
            ((of.x0 <= df.x0) & (df.x0 <= of.x1))
            | ((of.x0 <= df.x1) & (df.x1 <= of.x1))
            | ((df.x0 <= of.x0) & (of.x0 <= df.x1))
            | ((df.x0 <= of.x1) & (of.x1 <= df.x1))
        )
        df = df[bottom_mask].sort_values(sort_by)
        # self.caches['bottom_box'][of.name] = index
        return df.iloc[:limit]

    def find_invoice_num(self):
        _logger.debug("find_invoice_num start")
        left_bottom_bboxes = self.__find_left_and_bottom_box__(
            self.invoice_number_activation_token
        )

        extracted_invoice_numbers = left_bottom_bboxes.copy()
        extracted_invoice_numbers["invoice_number"] = (
            extracted_invoice_numbers.text.apply(
                lambda x: re.search(self.invoice_number_regex, x, re.IGNORECASE)
            ).apply(lambda x: x.group(0) if x is not None else np.nan)
        )
        extracted_invoice_numbers = extracted_invoice_numbers[
            ~extracted_invoice_numbers.invoice_number.isna()
        ]
        extracted_invoice_numbers = extracted_invoice_numbers.sort_values("y0")
        result = extracted_invoice_numbers.invoice_number.to_numpy()
        _logger.debug(result)
        _logger.debug("find_invoice_num end")
        return result[0] if len(result) > 0 else np.nan

    def find_date(self):
        _logger.debug("find_date start")
        dates = self.bboxes.text.str.extract(self.date_regex).date
        try:
            result_date = dates[~dates.isna()].iloc[0]
        except IndexError:
            result_date = np.nan
        _logger.debug("find_date end")
        return result_date

    def find_vat(self):
        _logger.debug("find_vat start")
        results = set()
        # matcher1 = re.compile(r".*\D(?P<vat>[0-9]+/?[A-Z])/?[A-Z]/?[A-Z]/?[0O]{3}.*")

        # matcher2 = re.compile(r".*\D(?P<vat>[0-9]{3,}\s?[A-Z]).*")
        for i, box in self.bboxes.iterrows():
            text = box.text.strip().replace(",", ".").replace(" ", "").replace(".", "")
            # match = re.search(r"(?P<vat>\d{3,}[ \WA-Z]+[0O ]{3,})", text, re.IGNORECASE)
            match = re.search(self.vat_regex, text, re.IGNORECASE)
            if not match and ratio_similarity(box, self.vat_activation_token) >= 1:
                # Essayer de façon horizontal
                df = self.__get_left_box__(box, limit=1)
                text = (
                    df.iloc[0].text.strip().replace(",", ".").replace(" ", "").upper()
                    if len(df) >= 1
                    else ""
                )
                match = re.match(self.vat_regex, text, re.IGNORECASE) if text else False
                if match:
                    continue
                # for j in range(len(df)):
                #     text = df.iloc[j].text.strip().replace(",", ".").replace(" ", "").upper()
                #     # print(text, end=" ===== ")
                #     match = matcher2.match(text)
                #     if match:
                #         break
                # else:
                # Essayer de façon vertical
                df = self.__get_bottom_box__(box, limit=1)
                text = (
                    df.iloc[0].text.strip().replace(",", ".").replace(" ", "").upper()
                    if len(df) >= 1
                    else ""
                )
                match = re.match(self.vat_regex, text, re.IGNORECASE) if text else False
            if match:
                vat = match.groupdict().get("vat", None)
                if vat:
                    results.add(vat)
        _logger.debug("find_vat end")
        return list(results)

    def find_amount(self):
        _logger.debug("find_amount start")
        results = {}
        matcher = re.compile(r".*?(?P<amount>\d+[.]\d{3}).*")
        for i, box in self.bboxes.iterrows():
            if i in self.invoice_line_header.index:
                continue
            if ratio_similarity(box, self.amounts_activation_token) < 1:
                continue
            # Recherche dans la boite et eviler les boite avec %
            text = box.text.strip().replace(",", ".").replace(" ", "")
            # if "%" not in text:
            #     match = matcher.match(text)
            # else:
            #     match = None
            match = matcher.match(text)
            if not match:
                # Essayer de façon horizontal
                df = self.__get_left_box__(box, limit=10)
                # Liste des box donc la ligne passant par leur centre est entre le segment vertical de la box
                df["is_in_vertical_segment"] = ((df.y0 + df.y1) / 2).between(
                    box.y0, box.y1
                )
                df = df[df.is_in_vertical_segment].sort_values(
                    by=["is_in_vertical_segment", "x0"]
                )
                match = None

                for j in range(len(df)):
                    text = df.iloc[j].text.strip().replace(",", ".").replace(" ", "")
                    match = matcher.match(text)
                    if match:
                        break
                else:
                    # Essayer de façon vertical
                    df = self.__get_bottom_box__(box, limit=1)
                    # Trier par le plus proche sur la ligne horizontal passant par le centre de la box ou verticalement
                    df["dist_from_vertical_line"] = np.abs(
                        df.c0 - box.c0
                    )  # Par rapport à l'axe vertical passant par le centre de la box
                    df = df.sort_values(by=["dist_from_vertical_line", "c1"])
                    if len(df) >= 1:
                        text = (
                            df.iloc[0].text.strip().replace(",", ".").replace(" ", "")
                        )
                        match = matcher.match(text)

            if match:
                results[box.text] = float(match["amount"])
        _logger.debug("find_amount end")
        return results

    def result(self, reload_token=True) -> pd.Series:
        if reload_token:
            self.__init_activation_token__()

        if self.data is None:
            self.__entities__()
        return self.data

    def __entities__(self):
        base_taxe = self.find_base_amount()
        amounts = self.find_amount()
        if amounts:
            self.data = pd.Series(
                {
                    "filename": self.filename,
                    "invoice_number": self.find_invoice_num(),
                    "invoice_date": self.find_date(),
                    "VAT": self.find_vat(),
                    "total_ht": base_taxe if base_taxe else get_total_ht(amounts),
                    "total_ttc": get_total_ttc(amounts),
                    "total_tva": get_total_tva(amounts),
                    "total_remise": get_total_remise(amounts),
                    "total_taxe": get_total_taxe(amounts),
                    "total_timbre": get_total_timbre(amounts),
                    "base_taxe": base_taxe,
                    "invoice_amounts": amounts,
                }
            )
        else:
            self.data = pd.Series(
                {
                    "filename": self.filename,
                    "invoice_number": self.find_invoice_num(),
                    "invoice_date": self.find_date(),
                    "VAT": self.find_vat(),
                    "total_ht": base_taxe,
                    "total_ttc": None,
                    "total_tva": None,
                    "total_remise": None,
                    "total_taxe": None,
                    "total_timbre": None,
                    "base_taxe": base_taxe,
                    "invoice_amounts": amounts,
                }
            )

    def _get_base_tax_line_header(self):
        """
        Recherche la ligne du tableau de taxe
        """
        list_of_token_already_in_header = ["base"]
        relative_tokens = ["code", "taux", "%", "tax", "montant", "tva", "mt", "t.v.a"]
        relative_token_df = pd.DataFrame(
            {"token": relative_tokens, "ratio": [1 for _ in relative_tokens]}
        )
        # Cherche tout les boxes contenant list_of_token_already_in_header
        # Filtre ceux dont la ligne ne contenant pas un relative_tokens
        # Récupère la boite la plus en bas du reste
        # Retourne tout les montants en dessous direct de la boite

        mask = self.bboxes.text.apply(
            lambda x: any(word in x.lower() for word in list_of_token_already_in_header)
        )
        filtered_boxes = self.bboxes[mask]
        max_index = 0
        df = None
        for i, box in filtered_boxes.iterrows():
            line_df = self.bboxes[
                self.bboxes.y0.between(box.y0, box.y1)
                | self.bboxes.y1.between(box.y0, box.y1)
            ]
            count = 0
            for _, box_line in line_df.iterrows():
                count += ratio_similarity(box_line, relative_token_df)
            if count > max_index:
                max_index = count
                df = line_df.copy()
        return df

    def find_base_amount(self):
        """
        Cherche le total ht (base de calcul de tax)
        """
        if self.base_tax_line_header is None:
            return None
        results = set()
        matcher = re.compile(r".*?(?P<amount>\d+[.]\d{3}).*")
        for _, box in self.base_tax_line_header.iterrows():
            if "base" in box.text.lower():
                # Recherche enregistre la boite du bas étant des montant jusqua rencontrer un qui n'est pas
                to_check_df = self.bboxes[
                    self.bboxes.text.str.contains(r"\d+[., ]\d{3}(\s|$)", regex=True)
                    & (self.bboxes.y1 > box.y1)
                    & (
                        self.bboxes.x0.between(box.x0, box.x1)
                        | self.bboxes.x1.between(box.x0, box.x1)
                    )
                ]

                for _, box1 in to_check_df.iterrows():
                    text = box1.text.strip().replace(",", ".").replace(" ", "")
                    match = matcher.match(text)
                    if match:
                        results.add(float(match["amount"]))
        return None if not results else max(results)


def main(image_dir: pathlib.Path) -> pd.Series:
    img_file = list(image_dir.glob("*.png"))[0]
    image = image_from_file(img_file)
    data_extractor = InvoiceDataExtractor(image=image, filename=img_file.stem)
    return data_extractor.result()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    image_dir = DATA_DIR / "test" / "images"
    main(image_dir)
