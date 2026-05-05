"""python -m singlet — print atlas summary and usage hints."""

import singlet


def main() -> None:
    print(f"singlet v{singlet.__version__}")
    print(singlet.summary())
    print()
    print("Quick start:")
    print("  import singlet")
    print('  singlet.catalog("lung")          # search datasets')
    print('  singlet.info("GSE264667")        # dataset metadata')
    print('  singlet.samples(tissue="brain")  # filter samples')
    print('  adata = singlet.load("GSM...")   # load → AnnData')
    print()
    print("For the MCP server:  python -m singlet.mcp")
    print("Debug info:          singlet.show_versions()")
    print("Full docs:           https://singlet.bio")


if __name__ == "__main__":
    main()
