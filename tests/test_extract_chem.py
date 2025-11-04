from qcatch.utils import parse_saved_chem


# Test cases for parse_saved_chem function
def test_parse_saved_chem_piscem_chromium_v3():
    """piscem mapping log with -g chromium_v3 should map to 10X_3p_v3"""
    map_info = {
        "cmdline": (
            "sc_ref_mapper -i /scratch0/rob/sc_data/refdata-gex-GRCh38-2024-A_piscem_index/index/piscem_idx "
            "-g chromium_v3 -1 R1.fastq.gz -2 R2.fastq.gz -t 32 "
            "-o /scratch0/rob/sc_data/af_map"
        ),
        "mapper": "piscem",
        "mode": "sc-rna",
    }
    assert parse_saved_chem(map_info) == "10X_3p_v3"


def test_parse_saved_chem_salmon_chromium_flag():
    """salmon cmdline containing --chromium should map to 10X_3p_v2"""
    map_info = {
        "cmdline": "salmon quant --chromium -i idx -l ISR",
        "mapper": "salmon",
    }
    assert parse_saved_chem(map_info) == "10X_3p_v2"


def test_parse_saved_chem_unknown_mapper_returns_none():
    """Unknown mapper values should be handled and return None"""
    map_info = {
        "cmdline": "somecmd --opt",
        "mapper": "mystery_mapper",
    }
    assert parse_saved_chem(map_info) is None
