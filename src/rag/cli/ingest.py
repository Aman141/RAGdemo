import argparse
from pathlib import Path

from rag.ingestion.pipeline import run_ingestion


def main() -> None:
	parser = argparse.ArgumentParser(description="Run PDF ingestion → chunks JSON")
	parser.add_argument("pdf", type=Path, help="Path to input PDF")
	parser.add_argument("output", type=Path, help="Path to output JSON")
	args = parser.parse_args()

	run_ingestion(str(args.pdf), str(args.output))


if __name__ == "__main__":
	main()


