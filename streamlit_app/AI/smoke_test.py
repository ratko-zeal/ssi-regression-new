"""Build data if needed and run a single end-to-end Responses call."""
import argparse
import json

from .builder import build, validate
from .handler import answer_question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="?", default="What is the final score for Kenya and its domain breakdown?")
    parser.add_argument("--session", type=str, default="local_session")
    args = parser.parse_args()

    # Build if needed
    try:
        validate()
    except Exception:
        build()

    out = answer_question(args.question, session_id=args.session, use_web=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
